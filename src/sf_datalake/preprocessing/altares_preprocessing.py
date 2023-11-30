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

import pyspark.sql.functions as F

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
    "--configuration",
    help="Configuration file containing required time-computations.",
    required=True,
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)
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

selected_cols = ["siren", "periode", "paydex", "fpi_30", "fpi_90"]

# Handle missing values and export
mvh = sf_datalake.transform.MissingValuesHandler(
    inputCols=["paydex", "fpi_30", "fpi_90"],
    value=configuration.preprocessing.fill_default_values,
)

sf_datalake.io.write_data(
    mvh.transform(df.select(selected_cols)),
    args.output,
    args.output_format,
)
