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

import pyspark.sql.functions as F
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
    "-c",
    "--configuration",
    help="Configuration file.",
    default="standard.json",
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)


args = parser.parse_args()
configuration = sf_datalake.config.ConfigurationHelper(args.configuration)
input_ds = sf_datalake.io.load_data(
    {"input": args.input}, file_format="csv", sep=",", infer_schema=False
)["input"]

# Set every column name to lower case (if not already).
siret_level_ds = input_ds.toDF(*(col.lower() for col in input_ds.columns))

# Cast 'periode' to a "beginning of the month" date.
siret_level_ds = siret_level_ds.withColumn(
    "periode", F.to_date(F.date_trunc("month", F.col("periode")))
)

#####################
# SIREN aggregation #
#####################

# Filter out public institutions and companies and aggregate at SIREN level
siren_converter = sf_datalake.transform.SiretToSiren()
aggregator = sf_datalake.transform.SirenAggregator(
    grouping_cols=configuration.preprocessing.identifiers,
    aggregation_map=configuration.preprocessing.siren_aggregation,
    no_aggregation=[],
)
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

# pylint: disable=unsubscriptable-object

time_computations: List[Transformer] = []
for feature, n_months in configuration.preprocessing.time_aggregation["lag"].items():
    if feature in siren_level_ds.columns:
        time_computations.append(
            sf_datalake.transform.LagOperator(
                inputCol=feature, n_months=n_months, bfill=True
            )
        )
for feature, n_months in configuration.preprocessing.time_aggregation["diff"].items():
    if feature in siren_level_ds.columns:
        time_computations.append(
            sf_datalake.transform.DiffOperator(
                inputCol=feature, n_months=n_months, bfill=True
            )
        )
for feature, n_months in configuration.preprocessing.time_aggregation[
    "moving_average"
].items():
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
