"""Altares historicized data preprocessing.

This script parses and extracts data from raw historicized files supplied by Altares. It
will output a dataset containing the following information:
- siren
- periode (date, first day of each month where data is available)
- paydex
- fpi30
- fpi90

It also computes time-aggregates required through configuration.

"""

import os
import sys
from os import path
from typing import List

import pyspark.sql.functions as F
from pyspark.ml import Transformer

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

spark = sf_datalake.utils.get_spark_session()

parser = sf_datalake.io.data_path_parser()
parser.description = "Extract and pre-process Altares data."
parser.add_argument(
    "-t",
    "--time_computations",
    help="Configuration file containing required time-computations.",
    default="time_series.json",
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

time_agg_config = sf_datalake.io.load_variables(args.time_computations)
df = spark.read.csv(
    args.input,
    sep=";",
    inferSchema=False,
    header=True,
)

## Extraction
df = df.withColumn(
    "periode",
    F.to_date(F.date_trunc("month", F.to_date(F.col("DATE_VALEUR"), "dd/MM/yyyy"))),
)
# We have to extract from a dumb string format.
df = df.withColumn(
    "paydex",
    F.regexp_extract(
        F.regexp_replace(F.col("PAYDEX"), r"(^Pas de retard)", "0"), r"(^\d+)", 1
    ).cast("float"),
)
df = df.withColumn("fpi_30", F.col("NOTE100_ALERTEUR_PLUS_30").cast("float"))
df = df.withColumn("fpi_90", F.col("NOTE100_ALERTEUR_PLUS_90_JOURS").cast("float"))
df = df.withColumnRenamed("SIREN", "siren")

## Pre-processing
time_computations: List[Transformer] = []

for feature, n_months in time_agg_config["LAG"].items():
    if feature in df.columns:
        time_computations.append(
            sf_datalake.transform.LagOperator(inputCol=feature, n_months=n_months)
        )
for feature, n_months in time_agg_config["DIFF"].items():
    if feature in df.columns:
        time_computations.append(
            sf_datalake.transform.DiffOperator(inputCol=feature, n_months=n_months)
        )
for feature, n_months in time_agg_config["MOVING_AVERAGE"].items():
    if feature in df.columns:
        time_computations.append(
            sf_datalake.transform.MovingAverage(inputCol=feature, n_months=n_months)
        )

selected_cols = []
for sel_col in ["siren", "periode", "paydex", "fpi_30", "fpi_90"]:
    selected_cols.extend(df_col for df_col in df.columns if df_col.startswith(sel_col))

sf_datalake.io.write_data(
    df.select(selected_cols),
    args.output,
    args.output_format,
)
