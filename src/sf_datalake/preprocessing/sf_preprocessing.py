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
parser.description = "Build a dataset with aggregated SIREN-level data and new time \
averaged/lagged variables."

parser.add_argument(
    "-t",
    "--time_computations",
    help="Configuration file containing required time-computations.",
    default="time_series.json",
)
parser.add_argument(
    "-a",
    "--aggregation",
    help="Configuration file with aggregation info.",
    default="aggregation.json",
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)


args = parser.parse_args()
time_comp_config = sf_datalake.io.load_variables(args.time_computations)
agg_config = sf_datalake.io.load_variables(args.aggregation)
input_ds = sf_datalake.io.load_data(
    {"input": args.input}, file_format="csv", sep=",", infer_schema=True
)["input"]

# Set every column name to lower case (if not already).
siret_level_ds = input_ds.toDF(*(col.lower() for col in input_ds.columns))

#####################
# SIREN aggregation #
#####################

# Filter out public institutions and companies and aggregate at SIREN level
siren_converter = sf_datalake.transform.SiretToSiren()
aggregator = sf_datalake.transform.SirenAggregator(agg_config)
siren_level_ds = (
    PipelineModel([siren_converter, aggregator])
    .transform(siret_level_ds)
    .select(
        [
            "periode",
            "siren",
            "apart_heures_consommees",
            "cotisation",
            "effectif",
            "montant_part_ouvriere",
            "montant_part_patronale",
        ]
    )
)

#####################
# Time Computations #
#####################

time_computations: List[Transformer] = []
for feature, n_months in time_comp_config["LAG"].items():
    if feature in siren_level_ds.columns:
        time_computations.append(
            sf_datalake.transform.LagOperator(inputCol=feature, n_months=n_months)
        )
for feature, n_months in time_comp_config["DIFF"].items():
    if feature in siren_level_ds.columns:
        time_computations.append(
            sf_datalake.transform.DiffOperator(inputCol=feature, n_months=n_months)
        )
for feature, n_months in time_comp_config["MOVING_AVERAGE"].items():
    if feature in siren_level_ds.columns:
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

output_ds = PipelineModel(time_computations + feature_engineering).transform(
    siren_level_ds
)
sf_datalake.io.write_data(output_ds, args.output, args.output_format)
