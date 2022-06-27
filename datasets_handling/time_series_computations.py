"""Adds new columns to dataset by computing averages, lags, etc. of existing variables.

An output dataset will be stored as split orc files under the chosen output directory.

USAGE
    python aggregate_to_siren.py <input_dataset_directory> <output_directory>

"""
import os
import re
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
    "Configuration file from which required computations will be deduced.",
)

args = parser.parse_args()
config = sf_datalake.io.load_variables(args.config)
input_ds = sf_datalake.io.load_data({"input": args.input}, file_format="orc")["input"]
input_ds = input_ds.toDF(
    *(col.lower() for col in input_ds.columns)
)  # Set every column name to lower case (if not already).


#####################
# Time Computations #
#####################

# Get lagged and moving average variables from a configuration file.
lag_re = re.compile(r"_lag\d+m")
ma_re = re.compile(r"_moy\d+m")
lag_computations = {}
ma_computations = {}

for feature in config["FEATURES"]:
    if lag_re.search(feature) is not None:
        match = lag_re.search(feature)
        feat = feature[: match.start()]
        duration = feature[match.start() + 3 : -1]
        lag_computations.setdefault(feat, []).append(duration)
    elif ma_re.search(feature) is not None:
        match = ma_re.search(feature)
        feat = feature[: match.start()]
        duration = feature[match.start() + 3 : -1]
        ma_computations.setdefault(feat, []).append(duration)

time_computations: List[Transformer] = []
for feat, n_months in lag_computations.items():
    time_computations.append(
        sf_datalake.transform.LagOperator(inputCol=feat, n_months=n_months)
    )
for feature, n_months in ma_computations.items():
    time_computations.append(
        sf_datalake.transform.MovingAverage(inputCol=feat, n_months=n_months)
    )

output_ds = PipelineModel(stages=time_computations).transform(input_ds)
output_ds.write.format("orc").save(args.output)
