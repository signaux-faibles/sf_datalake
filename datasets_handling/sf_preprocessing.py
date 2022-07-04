"""Carry out some pre-processing over the "sf" dataset.

Adds new columns to dataset by :
- computing averages, lags, etc. of existing variables.
- computing new features derived from existing ones.

An output dataset will be stored as split orc files under the chosen output directory.

USAGE
    python aggregate_to_siren.py <input_dataset_directory> <output_directory>

"""
import os
import sys
from os import path
from typing import List

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
parser.description = "Build a dataset with new time-transformed variables."
parser.add_argument(
    "-c",
    "--config",
    "Configuration file containing required computations.",
)

args = parser.parse_args()
config = sf_datalake.io.load_variables(args.config)
input_ds = sf_datalake.io.load_data({"input": args.input}, file_format="orc")["input"]
# Set every column name to lower case (if not already).
input_ds = input_ds.toDF(*(col.lower() for col in input_ds.columns))

#####################
# Time Computations #
#####################

time_computations: List[Transformer] = []
for feature, n_months in config["LAG"].items():
    time_computations.append(
        sf_datalake.transform.LagOperator(inputCol=feature, n_months=n_months)
    )
for feature, n_months in config["DIFF"].items():
    time_computations.append(
        sf_datalake.transform.DiffOperator(inputCol=feature, n_months=n_months)
    )
for feature, n_months in config["MOVING_AVERAGE"].items():
    time_computations.append(
        sf_datalake.transform.MovingAverage(inputCol=feature, n_months=n_months)
    )


#######################
# Feature engineering #
#######################

feature_engineering = [
    sf_datalake.transform.DebtRatioColumnAdder(),
    sf_datalake.transform.MovingAverage(inputCol="ratio_dette", n_months=12),
    sf_datalake.transform.DeltaDebtPerWorkforceColumnAdder(),
]

output_ds = PipelineModel(stages=time_computations + feature_engineering).transform(
    input_ds
)
output_ds.write.format("orc").save(args.output)
