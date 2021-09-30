import datetime
import os
from functools import reduce

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

spark = SparkSession.builder.getOrCreate()

### User-set parameters

FILL_MISSING_VALUES = True
OVERSAMPLING_RATIO = 0.2
REGULARIZATION_COEFF = 0.05
MAX_ITER = 50
TOL = 1e-5

#########
# Utils #
#########


def logging(msg):
    print(f"SF_LOG {datetime.datetime.utcnow()} {msg}")


def acoss_make_avg_delta_dette_par_effectif(data):
    """Compute the average change in social debt / nb of employees.

    Output column : 'avg_delta_dette_par_effectif'.
    """

    data = data.withColumn(
        "dette_par_effectif",
        (data["montant_part_ouvriere"] + data["montant_part_patronale"])
        / data["effectif"],
    )
    # TODO replace([np.nan, np.inf, -np.inf], 0)

    data = data.withColumn(
        "dette_par_effectif_past_3",
        (data["montant_part_ouvriere_past_3"] + data["montant_part_patronale_past_3"])
        / data["effectif"],
    )
    # TODO replace([np.nan, np.inf, -np.inf], 0)

    data = data.withColumn(
        "avg_delta_dette_par_effectif",
        (data["dette_par_effectif"] - data["dette_par_effectif_past_3"]) / 3,
    )

    columns_to_drop = ["dette_par_effectif", "dette_par_effectif_past_3"]
    return data.drop(*columns_to_drop)  # , axis=1, inplace=True)


def assemble_df(df, features_col):
    assembler = VectorAssembler(inputCols=features_col, outputCol="assembledFeatures")
    return assembler.transform(df)


def fit_to_data(assembled_df, sampling_ratio=None):
    std_scaler = StandardScaler(
        inputCol="assembledFeatures", outputCol="scaledFeatures"
    )
    if sampling_ratio is not None:
        scalerModel = std_scaler.fit(assembled_df.sample(sampling_ratio))
    else:
        scalerModel = std_scaler.fit(assembled_df)
    return scalerModel


def scale_df(scalerModel, df, obj_col, keep_cols=[]):
    scaledData = scalerModel.transform(df)
    return scaledData.select(
        keep_cols
        + [
            F.col("scaledFeatures").alias("features"),
            F.col(obj_col).cast("integer").alias("label"),
        ]
    )


## Variables definition

# CL2B recommended variables

MRV_VARIABLES = {
    "MNT_AF_BFONC_BFR",
    "MNT_AF_BFONC_TRESORERIE",
    "RTO_AF_RATIO_RENT_MBE",
    "MNT_AF_BFONC_FRNG",
    "MNT_AF_CA",
    "MNT_AF_SIG_EBE_RET",
    "RTO_AF_RENT_ECO",
    "RTO_AF_SOLIDITE_FINANCIERE",
    "RTO_INVEST_CA",
}

TAC_VARIABLES = {f"tac_1y_{v}" for v in MRV_VARIABLES}
MRV_VARIABLES.update(TAC_VARIABLES)

SUM_VARIABLES = {
    "cotisation",
    "cotisation_moy12m",
    "montant_part_ouvriere",
    "montant_part_ouvriere_past_1",
    "montant_part_ouvriere_past_12",
    "montant_part_ouvriere_past_2",
    "montant_part_ouvriere_past_3",
    "montant_part_ouvriere_past_6",
    "montant_part_patronale",
    "montant_part_patronale_past_1",
    "montant_part_patronale_past_12",
    "montant_part_patronale_past_2",
    "montant_part_patronale_past_3",
    "montant_part_patronale_past_6",
    "effectif",
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
}

AVG_VARIABLES = {
    "ratio_dette_moy12m",
}

COMP_VARIABLES = {
    "ratio_dette",
    "avg_delta_dette_par_effectif",
    # "paydex_nb_jours",
    # "paydex_nb_jours_past_12"
}

SF_VARIABLES = SUM_VARIABLES | AVG_VARIABLES | COMP_VARIABLES

BASE_VARIABLES = {
    "periode",
    "siren",
    "code_naf",
    "time_til_failure",
}
OBJ_VARIABLE = {"failure_within_18m"}
FEATURES = SF_VARIABLES | MRV_VARIABLES

TO_ONEHOT_ENCODE = set()
TO_SCALE = list(FEATURES - TO_ONEHOT_ENCODE)

logging(f"DGFiP Variables : {MRV_VARIABLES}")
logging(f"SF Variables : {SF_VARIABLES}")

ROOT_FOLDER = "/projets/TSF/sources/"
DATASET_PATH = "base/indicateurs_annuels.orc"
fullpath = os.path.join(ROOT_FOLDER, DATASET_PATH)
indics_annuels = spark.read.orc(fullpath)
logging(f"Reading data in {fullpath}")

### Default values for missing data

non_ratio_variables = list(filter(lambda x: x[:3] != "RTO", MRV_VARIABLES))
ratio_variables = list(filter(lambda x: x[:3] == "RTO", MRV_VARIABLES))

MRV_DEFAULT_DATA_VALUES = {
    v: 0.0 if v.find("tac") == -1 else 1.0 for v in non_ratio_variables
}

medians = reduce(
    lambda x, y: x + y, indics_annuels.approxQuantile(ratio_variables, [0.5], 0.05)
)

for var, med in zip(ratio_variables, medians):
    MRV_DEFAULT_DATA_VALUES[var] = med

SF_DEFAULT_DATA_VALUES = {
    "time_til_failure": 9999,
    ### ACOSS
    "montant_part_ouvriere_past_12": 0.0,
    "montant_part_patronale_past_12": 0.0,
    "montant_part_ouvriere_past_6": 0.0,
    "montant_part_patronale_past_6": 0.0,
    "montant_part_ouvriere_past_3": 0.0,
    "montant_part_patronale_past_3": 0.0,
    "montant_part_ouvriere_past_2": 0.0,
    "montant_part_patronale_past_2": 0.0,
    "montant_part_ouvriere_past_1": 0.0,
    "montant_part_patronale_past_1": 0.0,
    "cotisation": 0.0,
    "montant_part_ouvriere": 0.0,
    "montant_part_patronale": 0.0,
    "cotisation_moy12m": 0.0,
    "ratio_dette": 0.0,
    "ratio_dette_moy12m": 0.0,
    ### Activité partielle
    "apart_heures_autorisees": 0.0,
    "apart_heures_consommees_cumulees": 0.0,
    "apart_heures_consommees": 0.0,
    "avg_delta_dette_par_effectif": 0.0,
    ### Effectif
    "effectif": 0,
    "effectif_ent": 0,
}

DEFAULT_DATA_VALUES = dict(**MRV_DEFAULT_DATA_VALUES, **SF_DEFAULT_DATA_VALUES)

if FILL_MISSING_VALUES:
    logging("Filling missing values with default values.")
    logging(f"Defaults : {DEFAULT_DATA_VALUES}")

    indics_annuels = indics_annuels.fillna(
        {k: v for (k, v) in DEFAULT_DATA_VALUES.items() if k in indics_annuels.columns}
    )
else:
    indics_annuels.dropna(*FEATURES)


### Aggregation at SIREN level.

logging("Aggregating data at the SIREN level")

# Signaux faibles variables

indics_annuels_sf = indics_annuels.select(
    *(SUM_VARIABLES | AVG_VARIABLES | BASE_VARIABLES)
)

# Sums
gb_sum = indics_annuels_sf.groupBy("siren", "periode").sum(*SUM_VARIABLES)
for col_name in SUM_VARIABLES:
    gb_sum = gb_sum.withColumnRenamed(f"sum({col_name})", col_name)

# Averages
gb_avg = indics_annuels_sf.groupBy("siren", "periode").avg(*AVG_VARIABLES)
for col_name in AVG_VARIABLES:
    gb_avg = gb_avg.withColumnRenamed(f"avg({col_name})", col_name)

### TODO : ratio_dette_moyenne12m should be computed from the
### aggregated ratio_dette variable.
# w = indics_annuels_sf.groupBy("siren", F.window(df.periode - 365
# days, "365 days")).avg("ratio_dette")

# Joining grouped data
indics_annuels_sf = (
    indics_annuels_sf.drop(*(SUM_VARIABLES | AVG_VARIABLES))
    .join(gb_sum, on=["siren", "periode"])
    .join(gb_avg, on=["siren", "periode"])
)

### Feature engineering
logging("Feature engineering")

# delta_dette_par_effectif
indics_annuels_sf = acoss_make_avg_delta_dette_par_effectif(indics_annuels_sf)

# ratio_dette : real computation after sum
indics_annuels_sf = indics_annuels_sf.withColumn(
    "ratio_dette",
    (indics_annuels_sf.montant_part_ouvriere + indics_annuels_sf.montant_part_patronale)
    / indics_annuels_sf.cotisation_moy12m,
)

indics_annuels_sf = indics_annuels_sf.dropDuplicates(["siren", "periode"])

## TODO : paydex groups
# indics_annuels_sf = ...

# DGFIP Variables
indics_annuels_dgfip = indics_annuels.select(
    *(MRV_VARIABLES | {"siren", "periode"})
).dropDuplicates(["siren", "periode"])

# Joining data
indics_annuels = indics_annuels_sf.join(indics_annuels_dgfip, on=["siren", "periode"])

if FILL_MISSING_VALUES:
    indics_annuels = indics_annuels.fillna(
        {k: v for (k, v) in DEFAULT_DATA_VALUES.items() if k in indics_annuels.columns}
    )
else:
    indics_annuels.dropna(*FEATURES)

logging("Creating objective variable 'failure_within_18m'")
indics_annuels = indics_annuels.withColumn(
    "failure_within_18m", indics_annuels["time_til_failure"] <= 18
)

indics_annuels = indics_annuels.select(
    *(BASE_VARIABLES | FEATURES | OBJ_VARIABLE)
).filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")

logging("Filtering out firms on 'effectif' and 'code_naf' variables.")

### Learning

# Oversampling
logging(f"Creating oversampled training set ({OVERSAMPLING_RATIO})")

will_fail_mask = indics_annuels["failure_within_18m"]

n_samples = indics_annuels.count()
n_failing = indics_annuels.filter(will_fail_mask).count()
subset_size = int(n_failing / OVERSAMPLING_RATIO)
n_not_failing = int((1.0 - OVERSAMPLING_RATIO) * subset_size)

failing_subset = indics_annuels.filter(will_fail_mask)
not_failing_subset = indics_annuels.filter(~will_fail_mask).sample(
    n_not_failing / (n_samples - n_failing)
)
oversampled_subset = failing_subset.union(not_failing_subset)

# Define dates
TRAIN_DATES = ("2016-01-01", "2018-05-31")
TEST_DATES = ("2018-06-01", "2018-11-01")
PREDICTION_DATE = "2020-02-01"

SIREN_train, SIREN_test = (
    indics_annuels.select("siren").distinct().randomSplit([0.8, 0.2])
)

logging(f"Creating train set over {TRAIN_DATES}.")
train = (
    oversampled_subset.filter(oversampled_subset["siren"].isin(SIREN_train["siren"]))
    .filter(oversampled_subset["periode"] > TRAIN_DATES[0])
    .filter(oversampled_subset["periode"] < TRAIN_DATES[1])
)

logging(f"Creating test set over {TEST_DATES}.")
test = (
    indics_annuels.filter(indics_annuels["siren"].isin(SIREN_test["siren"]))
    .filter(indics_annuels["periode"] > TEST_DATES[0])
    .filter(indics_annuels["periode"] < TEST_DATES[1])
)

logging(f"Creating a prediction subset on {PREDICTION_DATE}.")
prediction = indics_annuels.filter(
    F.to_date(indics_annuels["periode"]) == PREDICTION_DATE
)

assembled_train = assemble_df(train, TO_SCALE)
assembled_test = assemble_df(test, TO_SCALE)
assembled_prediction = assemble_df(prediction, TO_SCALE)

scaler = fit_to_data(assembled_train)

scaled_train = scale_df(scaler, assembled_train, obj_col="failure_within_18m")
scaled_test = scale_df(scaler, assembled_test, obj_col="failure_within_18m")
scaled_prediction = scale_df(
    scaler,
    assembled_prediction,
    obj_col="failure_within_18m",
    keep_cols=["siren", "periode"],
)

# Training
logging(
    f"Training logistic regression model with regularization \
    {REGULARIZATION_COEFF} and {MAX_ITER} iterations (maximum)."
)
blor = LogisticRegression(
    regParam=REGULARIZATION_COEFF, standardization=False, maxIter=MAX_ITER, tol=TOL
)
blorModel = blor.fit(scaled_train)
logging(f"Model weights: {blorModel.coefficients}")
logging(f"Model intercept: {blorModel.intercept}")

# Failing probability extraction
positive_class_proba_extractor = F.udf(lambda v: float(v[1]), FloatType())

# Test data for optimal threshold computaion
logging("Running model on test dataset.")

test_data = blorModel.transform(scaled_test)
test_data = test_data.select(["probability", "label"])
test_data = test_data.withColumn(
    "positive_class_probability", positive_class_proba_extractor("probability")
).drop("probability")

# Prediction
logging("Running model on prediction dataset.")

prediction_data = blorModel.transform(scaled_prediction).drop(
    "rawPrediction", "label", "features"
)
prediction_data = prediction_data.withColumn(
    "positive_class_probability", positive_class_proba_extractor("probability")
).drop("probability")

# Write outputs to csv
base_output_path = "/projets/TSF/donnees/sorties_modeles/"
output_folder = os.path.join(
    base_output_path,
    "logreg{}_train{}to{}_test{}_to{}_predict{}/".format(
        REGULARIZATION_COEFF, *TRAIN_DATES, *TEST_DATES, PREDICTION_DATE
    ),
)
test_output_path = os.path.join(output_folder, "test_data/")
prediction_output_path = os.path.join(output_folder, "prediction_data/")

logging("Writing test data to file {}".format(test_output_path))
test_data.repartition(1).write.format("csv").save(test_output_path)

logging("Writing prediction data to file {}".format(prediction_output_path))
prediction_data.repartition(1).write.format("csv").save(prediction_output_path)
