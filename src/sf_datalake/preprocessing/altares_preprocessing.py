"""Altares data preprocessing.

This script parses and extracts data from raw historicized files supplied by Altares. It
will output a dataset containing the following information:
- siren
- période (date, first day of each month where data is available)
- paydex
- fpi_30
- fpi_90


The input file may be a concatenation of different files that are not sampled evenly
across time. In case there are multiple rows for a given (SIREN, month) couple, we only
keep the most recent data and truncate date to the first day of month.

"""

import os
import sys
from os import path

import pyspark.sql.functions as F
import pyspark.sql.types as T

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
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

paydex_schema = T.StructType(
    [
        T.StructField("siren", T.StringType(), False),
        T.StructField("état_organisation", T.StringType(), True),
        T.StructField("code_paydex", T.IntegerType(), True),
        T.StructField("paydex", T.FloatType(), True),
        T.StructField("n_fournisseurs", T.IntegerType(), True),
        T.StructField("encours_étudiés", T.FloatType(), True),
        T.StructField("fpi_30", T.FloatType(), True),
        T.StructField("fpi_90", T.FloatType(), True),
        T.StructField("date", T.DateType(), False),
    ]
)

df = spark.read.csv(args.input, sep=",", header=True, schema=paydex_schema)
num_cols = [
    "paydex",
    "fpi_30",
    "fpi_90",
    "encours_étudiés",
    "n_fournisseurs",
]

## Pre-processing and export

# Clip and normalize FPIs to the [0, 1] range
df = df.withColumn("fpi_30", sf_datalake.utils.clip("fpi_30", lower=0, upper=100) / 100)
df = df.withColumn("fpi_90", sf_datalake.utils.clip("fpi_90", lower=0, upper=100) / 100)

# There may be multiple data for a given month, in which case we only keep last values
df = df.groupBy(["siren", F.trunc(format="month", date="date").alias("période")]).agg(
    *(F.last(col).alias(col) for col in num_cols)
)

sf_datalake.io.write_data(
    df,
    args.output,
    args.output_format,
)
