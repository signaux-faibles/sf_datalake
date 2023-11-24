"""Carry out some pre-processing over the "sf" dataset.

1) Adds new columns to dataset by:
- computing averages, lags, etc. of existing variables.
- computing new features derived from existing ones.
2) Aggregates data at the SIREN level.

An output dataset will be stored as split orc files under the chosen output directory.

USAGE
    python sf_preprocessing.py <input_directory> <output_directory> \
-t [time_computations_config_filename] -a [aggregation_config_filename]

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

# pylint: disable=C0413
import sf_datalake.io
import sf_datalake.transform

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset with aggregated SIREN-level data and new time \
averaged/lagged variables."

parser.add_argument("-c", "--configuration", help="Configuration file.", required=True)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)


args = parser.parse_args()
configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)
input_ds = sf_datalake.io.load_data(
    {"input": args.input}, file_format="csv", sep=",", infer_schema=False
)["input"]

# Set every column name to lower case (if not already).
df = input_ds.toDF(*(col.lower() for col in input_ds.columns))

#####################
# Time Computations #
#####################

# pylint: disable=unsubscriptable-object

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
            stat_strategy=strategy,
        )
        for strategy, features in imputation_strategy_features.items()
    )

output_ds = PipelineModel(
    stages=time_computations + missing_values_handling_steps
).transform(df)


sf_datalake.io.write_data(output_ds, args.output, args.output_format)
