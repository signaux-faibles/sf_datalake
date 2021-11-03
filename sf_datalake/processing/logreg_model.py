import logging
import os
from functools import reduce

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import ElementwiseProduct
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType
from sf_datalake.configuration import config
from sf_datalake.utils import feature_engineering, preprocessing

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

ROOT_FOLDER = "/projets/TSF/sources/"
DATASET_PATH = "base/indicateurs_annuels.orc"
fullpath = os.path.join(ROOT_FOLDER, DATASET_PATH)
indics_annuels = spark.read.orc(fullpath)
logging.info(f"Reading data in {fullpath}")

### Default values for missing data

non_ratio_variables = list(filter(lambda x: x[:3] != "RTO", config.MRV_VARIABLES))
ratio_variables = list(filter(lambda x: x[:3] == "RTO", config.MRV_VARIABLES))

mrv_default_data_values = {v: 0.0 for v in non_ratio_variables}

medians = reduce(
    lambda x, y: x + y, indics_annuels.approxQuantile(ratio_variables, [0.5], 0.05)
)

for var, med in zip(ratio_variables, medians):
    mrv_default_data_values[var] = med


default_data_values = dict(**mrv_default_data_values, **config.SF_DEFAULT_DATA_VALUES)

if config.FILL_MISSING_VALUES:
    logging.info("Filling missing values with default values.")
    logging.info(f"Defaults : {default_data_values}")

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
    *(config.SUM_VARIABLES | config.AVG_VARIABLES | config.BASE_VARIABLES)
)

# Sums
gb_sum = indics_annuels_sf.groupBy("siren", "periode").sum(*config.SUM_VARIABLES)
for col_name in config.SUM_VARIABLES:
    gb_sum = gb_sum.withColumnRenamed(f"sum({col_name})", col_name)

# Averages
gb_avg = indics_annuels_sf.groupBy("siren", "periode").avg(*config.AVG_VARIABLES)
for col_name in config.AVG_VARIABLES:
    gb_avg = gb_avg.withColumnRenamed(f"avg({col_name})", col_name)

### TODO : ratio_dette_moyenne12m should be computed from the
### aggregated ratio_dette variable.
# w = indics_annuels_sf.groupBy("siren", F.window(df.periode - 365
# days, "365 days")).avg("ratio_dette")

# Joining grouped data
indics_annuels_sf = (
    indics_annuels_sf.drop(*(config.SUM_VARIABLES | config.AVG_VARIABLES))
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

## Paydex
indics_annuels_paydex = indics_annuels.select(
    *(config.PAYDEX_VARIABLES | {"siren", "periode"})
).dropDuplicates(["siren", "periode"])
indics_annuels_paydex = feature_engineering.make_paydex_bins(indics_annuels_paydex)
indics_annuels_paydex = feature_engineering.make_paydex_yoy(indics_annuels_paydex)

# DGFIP Variables
indics_annuels_dgfip = indics_annuels.select(
    *(config.MRV_VARIABLES | {"siren", "periode"})
).dropDuplicates(["siren", "periode"])

# Joining data
indics_annuels = indics_annuels_sf.join(
    indics_annuels_dgfip, on=["siren", "periode"]
).join

if config.FILL_MISSING_VALUES:
    indics_annuels = indics_annuels.fillna(
        {
            k: v
            for (k, v) in config.DEFAULT_DATA_VALUES.items()
            if k in indics_annuels.columns
        }
    )
else:
    indics_annuels = indics_annuels.dropna(subset=tuple(config.FEATURES))

logging.info("Creating objective variable 'failure_within_18m'")
indics_annuels = indics_annuels.withColumn(
    "failure_within_18m", indics_annuels["time_til_failure"] <= 18
)

indics_annuels = indics_annuels.select(
    *(config.BASE_VARIABLES | config.FEATURES | config.OBJ_VARIABLE)
).filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")

logging.info("Filtering out firms on 'effectif' and 'code_naf' variables.")

### Learning

# Oversampling
logging.info(f"Creating oversampled training set ({config.OVERSAMPLING_RATIO})")

will_fail_mask = indics_annuels["failure_within_18m"]

n_samples = indics_annuels.count()
n_failing = indics_annuels.filter(will_fail_mask).count()
subset_size = int(n_failing / config.OVERSAMPLING_RATIO)
n_not_failing = int((1.0 - config.OVERSAMPLING_RATIO) * subset_size)

failing_subset = indics_annuels.filter(will_fail_mask)
not_failing_subset = indics_annuels.filter(~will_fail_mask).sample(
    n_not_failing / (n_samples - n_failing)
)
oversampled_subset = failing_subset.union(not_failing_subset)

# Define dates

SIREN_train, SIREN_test = (
    indics_annuels.select("siren").distinct().randomSplit([0.8, 0.2])
)

logging.info(f"Creating train set over {config.TRAIN_DATES}.")
train = (
    oversampled_subset.filter(oversampled_subset["siren"].isin(SIREN_train["siren"]))
    .filter(oversampled_subset["periode"] > config.TRAIN_DATES[0])
    .filter(oversampled_subset["periode"] < config.TRAIN_DATES[1])
)

logging.info(f"Creating test set over {config.TEST_DATES}.")
test = (
    indics_annuels.filter(indics_annuels["siren"].isin(SIREN_test["siren"]))
    .filter(indics_annuels["periode"] > config.TEST_DATES[0])
    .filter(indics_annuels["periode"] < config.TEST_DATES[1])
)

logging.info(f"Creating a prediction subset on {config.PREDICTION_DATE}.")
prediction = indics_annuels.filter(
    F.to_date(indics_annuels["periode"]) == config.PREDICTION_DATE
)

assembled_std_train = preprocessing.assemble_features(
    train, config.STD_SCALE_FEATURES, "assembled_std_features"
)
assembled_std_test = preprocessing.assemble_features(
    test, config.STD_SCALE_FEATURES, "assembled_std_features"
)
assembled_std_prediction = preprocessing.assemble_features(
    prediction, config.STD_SCALE_FEATURES, "assembled_std_features"
)
assembled_onehot_train = preprocessing.assemble_features(train, config.ONEHOT_FEATURES)
assembled_onehot_test = preprocessing.assemble_features(test, config.ONEHOT_FEATURES)
assembled_onehot_prediction = preprocessing.assemble_features(
    prediction, config.ONEHOT_FEATURES
)

standard_scaler_model = preprocessing.fit_scaler(
    assembled_std_train,
    scaler_type="standard",
    input_colname="assembled_std_features",
    output_colname="std_scaled_features",
)

scaled_train = preprocessing.scale_df(assembled_std_train)
scaled_test = preprocessing.scale_df(
    keep_cols=["siren", "time_til_failure"],
)
scaled_prediction = preprocessing.scale_df(
    keep_cols=["siren"],
)

# Training
logging.info(
    f"Training logistic regression model with regularization \
    {config.REGULARIZATION_COEFF} and {config.MAX_ITER} iterations (maximum)."
)
blor = LogisticRegression(
    regParam=config.REGULARIZATION_COEFF,
    standardization=False,
    maxIter=config.MAX_ITER,
    tol=config.TOL,
)
blorModel = blor.fit(scaled_train)
w = blorModel.coefficients
b = blorModel.intercept
logging.info(f"Model weights: {w}")
logging.info(f"Model intercept: {b}")

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
    feature for features in config.FEATURE_GROUPS.values() for feature in features
]

# Get feature influence
ep = ElementwiseProduct()
ep.setScalingVec(w)
ep.setInputCol("features")
ep.setOutputCol("eprod")

explanation_df = (
    ep.transform(scaled_prediction)
    .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
    .toDF(["siren"] + config.TO_SCALE)
)
for group, features in config.MESO_URSSAF_GROUPS.items():
    explanation_df = explanation_df.withColumn(
        group, sum(explanation_df[col] for col in features)
    ).drop(*features)

# 'Macro' scores per group
for group, features in config.FEATURE_GROUPS.items():
    explanation_df = explanation_df.withColumn(
        "{}_macro_score".format(group),
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
def get_max_value_colname(r, max_col):
    for i, name in enumerate(meso_features):
        if r[i] == max_col:
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
base_output_path = "/projets/TSF/donnees/sorties_modeles"
output_folder = os.path.join(
    base_output_path,
    "logreg{}_train{}to{}_test{}_to{}_predict{}".format(
        config.REGULARIZATION_COEFF,
        *config.TRAIN_DATES,
        *config.TEST_DATES,
        config.PREDICTION_DATE,
    ),
)
test_output_path = os.path.join(output_folder, "test_data")
prediction_output_path = os.path.join(output_folder, "prediction_data")
concerning_output_path = os.path.join(output_folder, "concerning_values")
explanation_output_path = os.path.join(output_folder, "explanation_data")

logging.info("Writing test data to file {}".format(test_output_path))
test_data.repartition(1).write.csv(test_output_path, header=True)

logging.info("Writing prediction data to file {}".format(prediction_output_path))
prediction_data.drop("features").repartition(1).write.csv(
    prediction_output_path, header=True
)

logging.info("Writing concerning features to file {}".format(concerning_output_path))
explanation_df.select(["siren"] + concerning_columns).repartition(1).write.csv(
    concerning_output_path, header=True
)

logging.info(
    "Writing explanation macro scores data to directory {}".format(
        explanation_output_path
    )
)
explanation_df.select(["siren"] + macro_scores_columns).repartition(1).write.csv(
    os.path.join(explanation_output_path), header=True
)
