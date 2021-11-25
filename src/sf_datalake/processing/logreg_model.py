"""Logistic regression model for company failure prediction.
"""

import datetime
import logging
import os
import sys
from functools import reduce
from os import path

from pyspark.ml.classification import LogisticRegression  # pylint: disable=E0401
from pyspark.ml.feature import ElementwiseProduct  # pylint: disable=E0401
from pyspark.sql import SparkSession  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401
from pyspark.sql.types import FloatType, StringType  # pylint: disable=E0401

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
from sf_datalake.config import base as model_config
from sf_datalake.preprocessing import (
    DATA_ROOT_DIR,
    OUTPUT_ROOT_DIR,
    feature_engineering,
)
from sf_datalake.processing import transform
from sf_datalake.utils import load_data

### Launch spark session, load data

spark = SparkSession.builder.getOrCreate()

logging.info(
    "Reading data in %s", path.join(DATA_ROOT_DIR, "base/indicateurs_annuels.orc")
)

indics_annuels = load_data(
    [("indics_annuels", "base/indicateurs_annuels.orc", DATA_ROOT_DIR)]
)["indics_annuels"]

### Default values for missing data

non_ratio_variables = list(filter(lambda x: x[:3] != "RTO", model_config.MRV_VARIABLES))
ratio_variables = list(filter(lambda x: x[:3] == "RTO", model_config.MRV_VARIABLES))

mrv_default_values = {v: 0.0 for v in non_ratio_variables}
data_medians = reduce(
    lambda x, y: x + y, indics_annuels.approxQuantile(ratio_variables, [0.5], 0.05)
)
for var, med in zip(ratio_variables, data_medians):
    mrv_default_values[var] = med


default_data_values = dict(**mrv_default_values, **model_config.SF_DEFAULT_VALUES)

if model_config.FILL_MISSING_VALUES:
    logging.info("Filling missing values with default values.")
    logging.info("Defaults : %s", default_data_values)

    indics_annuels = indics_annuels.fillna(
        {k: v for (k, v) in default_data_values.items() if k in indics_annuels.columns}
    )
else:
    indics_annuels = indics_annuels.fillna(
        {
            "time_til_failure": 9999,
        }
    )

### Aggregation at SIREN level.

logging.info("Aggregating data at the SIREN level")

# Signaux faibles variables
indics_annuels_sf = indics_annuels.select(
    *(
        model_config.SUM_VARIABLES
        | model_config.AVG_VARIABLES
        | model_config.BASE_VARIABLES
    )
)

# Sums
gb_sum = indics_annuels_sf.groupBy("siren", "periode").sum(*model_config.SUM_VARIABLES)
for col_name in model_config.SUM_VARIABLES:
    gb_sum = gb_sum.withColumnRenamed(f"sum({col_name})", col_name)

# Averages
gb_avg = indics_annuels_sf.groupBy("siren", "periode").avg(*model_config.AVG_VARIABLES)
for col_name in model_config.AVG_VARIABLES:
    gb_avg = gb_avg.withColumnRenamed(f"avg({col_name})", col_name)

### TODO : ratio_dette_moyenne12m should be computed from the
### aggregated ratio_dette variable.
# w = indics_annuels_sf.groupBy("siren", F.window(df.periode - 365
# days, "365 days")).avg("ratio_dette")

# Joining grouped data
indics_annuels_sf = (
    indics_annuels_sf.drop(*(model_config.SUM_VARIABLES | model_config.AVG_VARIABLES))
    .join(gb_sum, on=["siren", "periode"])
    .join(gb_avg, on=["siren", "periode"])
)

### Feature engineering

logging.info("Feature engineering")

# delta_dette_par_effectif
indics_annuels_sf = feature_engineering.avg_delta_debt_per_size(indics_annuels_sf)

# ratio_dette : real computation after sum
indics_annuels_sf = indics_annuels_sf.withColumn(
    "ratio_dette",
    (indics_annuels_sf.montant_part_ouvriere + indics_annuels_sf.montant_part_patronale)
    / indics_annuels_sf.cotisation_moy12m,
)

indics_annuels_sf = indics_annuels_sf.dropDuplicates(["siren", "periode"])

# DGFIP Variables
indics_annuels_dgfip = indics_annuels.select(
    *(model_config.MRV_VARIABLES | {"siren", "periode"})
).dropDuplicates(["siren", "periode"])

# Joining data
indics_annuels = indics_annuels_sf.join(indics_annuels_dgfip, on=["siren", "periode"])

if model_config.FILL_MISSING_VALUES:
    indics_annuels = indics_annuels.fillna(
        {k: v for (k, v) in default_data_values.items() if k in indics_annuels.columns}
    )
else:
    indics_annuels = indics_annuels.dropna(subset=tuple(model_config.FEATURES))

logging.info("Creating objective variable 'failure_within_18m'")
indics_annuels = indics_annuels.withColumn(
    "failure_within_18m", indics_annuels["time_til_failure"] <= 18
)

logging.info("Filtering out firms on 'effectif' and 'code_naf' variables.")
indics_annuels = indics_annuels.select(
    *(
        model_config.BASE_VARIABLES
        | model_config.FEATURES
        | model_config.TARGET_VARIABLE
    )
).filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")


### Learning

# Oversampling
logging.info(
    "Creating oversampled training set with positive examples ratio %.1f",
    model_config.OVERSAMPLING_RATIO,
)

will_fail_mask = indics_annuels["failure_within_18m"]

n_samples = indics_annuels.count()
n_failing = indics_annuels.filter(will_fail_mask).count()
subset_size = int(n_failing / model_config.OVERSAMPLING_RATIO)
n_not_failing = int((1.0 - model_config.OVERSAMPLING_RATIO) * subset_size)

failing_subset = indics_annuels.filter(will_fail_mask)
not_failing_subset = indics_annuels.filter(~will_fail_mask).sample(
    n_not_failing / (n_samples - n_failing)
)
oversampled_subset = failing_subset.union(not_failing_subset)

# Define dates
SIREN_train, SIREN_test = (
    indics_annuels.select("siren").distinct().randomSplit([0.8, 0.2])
)

logging.info("Creating train between %s and %s.", *model_config.TRAIN_DATES)
train = (
    oversampled_subset.filter(oversampled_subset["siren"].isin(SIREN_train["siren"]))
    .filter(oversampled_subset["periode"] > model_config.TRAIN_DATES[0])
    .filter(oversampled_subset["periode"] < model_config.TRAIN_DATES[1])
)

logging.info("Creating test set between %s and %s.", *model_config.TEST_DATES)
test = (
    indics_annuels.filter(indics_annuels["siren"].isin(SIREN_test["siren"]))
    .filter(indics_annuels["periode"] > model_config.TEST_DATES[0])
    .filter(indics_annuels["periode"] < model_config.TEST_DATES[1])
)

logging.info("Creating a prediction set on %s.", model_config.PREDICTION_DATE)
prediction = indics_annuels.filter(
    F.to_date(indics_annuels["periode"]) == model_config.PREDICTION_DATE
)

assembled_std_train = transform.assemble_features(
    train, model_config.STD_SCALE_FEATURES, "assembled_std_features"
)
assembled_std_test = transform.assemble_features(
    test, model_config.STD_SCALE_FEATURES, "assembled_std_features"
)
assembled_std_prediction = transform.assemble_features(
    prediction, model_config.STD_SCALE_FEATURES, "assembled_std_features"
)

standard_scaler_model = transform.fit_scaler(
    assembled_std_train,
    scaler_type="standard",
    input_colname="assembled_std_features",
    output_colname="std_scaled_features",
)

scaled_train = transform.scale_df(
    scaler_model=standard_scaler_model,
    df=assembled_std_train,
    features_col="std_scaled_features",
    label_col="failure_within_18m",
)
scaled_test = transform.scale_df(
    scaler_model=standard_scaler_model,
    df=assembled_std_test,
    features_col="std_scaled_features",
    label_col="failure_within_18m",
    keep_cols=["siren", "time_til_failure"],
)
scaled_prediction = transform.scale_df(
    scaler_model=standard_scaler_model,
    df=assembled_std_prediction,
    features_col="std_scaled_features",
    label_col="failure_within_18m",
    keep_cols=["siren"],
)

# Training
logging.info(
    "Training logistic regression model with regularization \
     %.3f and %d iterations (maximum).",
    model_config.REGULARIZATION_COEFF,
    model_config.MAX_ITER,
)
blor = LogisticRegression(
    regParam=model_config.REGULARIZATION_COEFF,
    standardization=False,
    maxIter=model_config.MAX_ITER,
    tol=model_config.TOL,
)
blorModel = blor.fit(scaled_train)
w = blorModel.coefficients
b = blorModel.intercept
logging.info("Model weights: %.3f", w)
logging.info("Model intercept: %.3f", b)

# Test data for optimal threshold computation
logging.info("Running model on test dataset.")

positive_class_proba_extractor = F.udf(lambda v: float(v[1]), FloatType())

test_data = blorModel.transform(scaled_test)
test_data = test_data.select(["siren", "time_til_failure", "label", "probability"])
test_data = test_data.withColumn(
    "probability", positive_class_proba_extractor("probability")
)

# Prediction
logging.info("Running model on prediction dataset.")
prediction_data = (
    blorModel.transform(scaled_prediction)
    .drop(
        "prediction",
        "rawPrediction",
        "label",
    )
    .withColumn("probability", positive_class_proba_extractor("probability"))
)

# Explain prediction
meso_features = [
    feature for features in model_config.FEATURE_GROUPS.values() for feature in features
]

# Get feature influence
ep = ElementwiseProduct()
ep.setScalingVec(w)
ep.setInputCol("features")
ep.setOutputCol("eprod")

explanation_df = (
    ep.transform(scaled_prediction)
    .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
    .toDF(["siren"] + model_config.STD_SCALE_FEATURES)
)
for group, features in model_config.MESO_URSSAF_GROUPS.items():
    explanation_df = explanation_df.withColumn(
        group, sum(explanation_df[col] for col in features)
    ).drop(*features)

# 'Macro' scores per group
for group, features in model_config.FEATURE_GROUPS.items():
    explanation_df = explanation_df.withColumn(
        f"{group}_macro_score",
        1 / (1 + F.exp(-(sum(explanation_df[col] for col in features)))),
    )
macro_scores_columns = [
    "sante_financiere_macro_score",
    "dette_urssaf_macro_score",
    "activite_partielle_macro_score",
    "misc_macro_score",
]

# 'Micro' scores
@F.udf(returnType=StringType())
def get_max_value_colname(row, max_col):
    """Extract the column name where the nth highest value is found."""
    for i, name in enumerate(meso_features):
        if row[i] == max_col:
            return name
    raise NameError(f"Could not find columns associated to {max_col}")


explanation_df = (
    explanation_df.withColumn(
        "1st_concerning_val",
        F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[0],
    )
    .withColumn(
        "2nd_concerning_val",
        F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[1],
    )
    .withColumn(
        "3rd_concerning_val",
        F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[2],
    )
)

explanation_df = (
    explanation_df.withColumn(
        "1st_concerning_feat",
        get_max_value_colname(F.array(meso_features), "1st_concerning_val"),
    )
    .withColumn(
        "2nd_concerning_feat",
        get_max_value_colname(F.array(meso_features), "2nd_concerning_val"),
    )
    .withColumn(
        "3rd_concerning_feat",
        get_max_value_colname(F.array(meso_features), "3rd_concerning_val"),
    )
)
concerning_columns = [
    "1st_concerning_val",
    "2nd_concerning_val",
    "3rd_concerning_val",
    "1st_concerning_feat",
    "2nd_concerning_feat",
    "3rd_concerning_feat",
]

# Write outputs to csv
base_output_path = path.join(OUTPUT_ROOT_DIR, "sorties_modeles")
output_folder = path.join(base_output_path, datetime.date.today().isoformat())
test_output_path = path.join(output_folder, "test_data")
prediction_output_path = path.join(output_folder, "prediction_data")
concerning_output_path = path.join(output_folder, "concerning_values")
explanation_output_path = path.join(output_folder, "explanation_data")

logging.info("Writing test data to file %s", test_output_path)
test_data.repartition(1).write.csv(test_output_path, header=True)

logging.info("Writing prediction data to file %s", prediction_output_path)
prediction_data.drop("features").repartition(1).write.csv(
    prediction_output_path, header=True
)

logging.info("Writing concerning features to file %s", concerning_output_path)
explanation_df.select(["siren"] + concerning_columns).repartition(1).write.csv(
    concerning_output_path, header=True
)

logging.info(
    "Writing explanation macro scores data to directory %s", explanation_output_path
)
explanation_df.select(["siren"] + macro_scores_columns).repartition(1).write.csv(
    path.join(explanation_output_path), header=True
)
