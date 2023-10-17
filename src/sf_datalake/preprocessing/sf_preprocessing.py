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
    PipelineModel(stages=[siren_converter, aggregator])
    .fit(siret_level_ds)
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
for feature, n_months in configuration.preprocessing.time_aggregation["mean"].items():
    if feature in siren_level_ds.columns:
        time_computations.append(
            sf_datalake.transform.MovingAverage(inputCol=feature, n_months=n_months)
        )

time_agg_ds = PipelineModel(stages=time_computations).transform(siren_level_ds)

#######################
# Feature engineering #
#######################

# Add "debt / contribution"
feat_eng_ds = time_agg_ds.withColumn(
    "dette_sur_cotisation_lissée",
    (time_agg_ds["montant_part_ouvriere"] + time_agg_ds["montant_part_patronale"])
    / time_agg_ds["cotisation_mean12m"],
)
# Moving average
feat_eng_ds = sf_datalake.transform.MovingAverage(
    inputCol="dette_sur_cotisation_lissée", n_months=12
).transform(feat_eng_ds)
# Average debt / workforce
feat_eng_ds = feat_eng_ds.withColumn(
    "dette_par_effectif",
    (feat_eng_ds["montant_part_ouvriere"] + feat_eng_ds["montant_part_patronale"])
    / feat_eng_ds["effectif"],
)
output_ds = sf_datalake.transform.DiffOperator(
    inputCol="dette_par_effectif",
    n_months=3,
    slope=True,
).transform(feat_eng_ds)


sf_datalake.io.write_data(output_ds, args.output, args.output_format)
