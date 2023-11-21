"""Carry out some pre-processing over URSSAF "cotisation" data.

USAGE
    python preprocess_urssaf.py <input_directory> <output_directory>

"""
# pylint: disable=duplicate-code
import datetime as dt
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
import pandas as pd

import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

####################
# Loading datasets #
####################

spark = sf_datalake.utils.get_spark_session()
parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset with aggregated SIREN-level data."
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
parser.add_argument("--min_date", default="2014-01-01")
args = parser.parse_args()

cotisation_schema = T.StructType(
    [
        T.StructField("siret", T.StringType(), False),
        T.StructField("numéro_compte", T.StringType(), True),
        T.StructField("fenêtre", T.StringType(), True),
        T.StructField("encaissé", T.DoubleType(), True),
        T.StructField("dû", T.DoubleType(), True),
    ]
)
siret_to_siren = sf_datalake.transform.SiretToSiren()

# Create a monthly date range that will become the time index
dr = pd.date_range(args.min_date, dt.date.today().isoformat(), freq="MS")
date_range = spark.createDataFrame(
    pd.DataFrame(dr.to_series().dt.date, columns=["période"])
)

## "Cotisation" data
cotisation = spark.read.csv(args.input, header=True, schema=cotisation_schema)
cotisation = cotisation.dropna(subset="fenêtre")
cotisation = cotisation.withColumn(
    "date_début", F.to_date(F.substring(F.col("fenêtre"), 1, 10))
)
cotisation = cotisation.withColumn(
    "date_fin", F.to_date(F.substring(F.col("fenêtre"), 21, 10))
)
cotisation = cotisation.filter(F.col("date_fin") > args.min_date)
cotisation = siret_to_siren.transform(cotisation)
cotisation = cotisation.withColumn(
    "cotisation_appelée_par_mois",
    F.col("dû") / F.months_between("date_fin", "date_début"),
)

cotisation = cotisation.join(
    date_range,
    on=date_range["période"].between(
        cotisation["date_début"], F.date_sub(cotisation["date_fin"], 1)
    ),
    how="inner",
)
output_ds = cotisation.groupBy(["siren", "période"]).agg(
    F.sum("cotisation_appelée_par_mois").alias("cotisation")
)
sf_datalake.io.write_data(output_ds, args.output, args.output_format)
