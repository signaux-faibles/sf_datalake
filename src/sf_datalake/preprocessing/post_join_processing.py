"""Carry out some pre-processing over the "join" dataset.

1) Manage Missing values
2) Adds new columns to dataset by:
- create target from judgment data;
- computing averages, lags, etc. of existing variables.

An output dataset will be stored as split orc files under the chosen output directory.

USAGE
    python post_join_processing.py <input_dataset> <output_dataset> \
-c [config_filename]

"""
import os
import sys
from os import path
from typing import Dict, List

from pyspark.ml import PipelineModel, Transformer

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=unsubscriptable-object, wrong-import-position
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a complete dataset with new time averaged/lagged variables."
parser.add_argument("-c", "--configuration", help="Configuration file.", required=True)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)


args = parser.parse_args()
configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)
spark = sf_datalake.utils.get_spark_session()
input_ds = spark.read.orc(args.input)

# Set every column name to lower case (if not already).
df = input_ds.toDF(*(col.lower() for col in input_ds.columns))


###################
# Target creation #
###################

labeling_step = [
    sf_datalake.transform.TargetVariable(
        inputCol=configuration.learning.target["judgment_date_col"],
        outputCol=configuration.learning.target["class_col"],
        n_months=configuration.learning.target["n_months"],
    ),
]

#######################
# Feature engineering #
#######################

df = df.withColumn(
    "dette_par_effectif",
    (df["dette_sociale_ouvrière"] + df["dette_sociale_patronale"]) / df["effectif"],
)

##########################
# Missing Value Handling #
##########################

missing_values_handling_steps = []
if configuration.preprocessing.fill_default_values:
    missing_values_handling_steps.append(
        sf_datalake.transform.MissingValuesHandler(
            inputCols=list(configuration.preprocessing.fill_default_values),
            value=configuration.preprocessing.fill_default_values,
        ),
    )
if configuration.preprocessing.fill_imputation_strategy:
    imputation_strategy_features: Dict[str, List[str]] = {}
    for (
        feature,
        strategy,
    ) in configuration.preprocessing.fill_imputation_strategy.items():
        imputation_strategy_features.setdefault(strategy, []).append(feature)

    missing_values_handling_steps.extend(
        sf_datalake.transform.MissingValuesHandler(
            inputCols=features,
            strategy=strategy,
        )
        for strategy, features in imputation_strategy_features.items()
    )

#####################
# Time computations #
#####################

time_computations: List[Transformer] = []
for feature, n_months in configuration.preprocessing.time_aggregation["lag"].items():
    time_computations.append(
        sf_datalake.transform.LagOperator(inputCol=feature, n_months=n_months)
    )
for feature, n_months in configuration.preprocessing.time_aggregation["diff"].items():
    time_computations.append(
        sf_datalake.transform.DiffOperator(inputCol=feature, n_months=n_months)
    )
for feature, n_months in configuration.preprocessing.time_aggregation["mean"].items():
    time_computations.append(
        sf_datalake.transform.MovingAverage(inputCol=feature, n_months=n_months)
    )

# Bfill after time computation
features_lag_bfill = [
    f"{feature}_lag{n_months}m"
    for feature, n_months_list in configuration.preprocessing.time_aggregation[
        "lag"
    ].items()
    for n_months in n_months_list
]

features_diff_bfill = [
    f"{feature}_diff{n_months}m"
    for feature, n_months_list in configuration.preprocessing.time_aggregation[
        "diff"
    ].items()
    for n_months in n_months_list
]

time_computations.append(
    sf_datalake.transform.MissingValuesHandler(
        inputCols=features_diff_bfill + features_lag_bfill, strategy="bfill"
    )
)


df = PipelineModel(
    stages=labeling_step + missing_values_handling_steps + time_computations
).transform(df)

## Feature engineering based on time computations
for n_months in configuration.preprocessing.time_aggregation.get("mean", {}).get(
    "cotisation", []
):
    df = df.withColumn(
        f"dette_sur_[cotisation_mean{n_months}m]",
        (df["dette_sociale_patronale"] + df["dette_sociale_ouvrière"])
        / df[f"cotisation_mean{n_months}m"],
    )

sf_datalake.io.write_data(df, args.output, args.output_format)
