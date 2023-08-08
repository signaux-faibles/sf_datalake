"""Extract activity period of companines

USAGE
    python sirene_activity.py -- <output_directory>
"""
import argparse
import os
import sys
from os import path

import pyspark.sql.functions as F

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413

import sf_datalake.utils

parser = argparse.ArgumentParser(description="Filter a dataset using companies activity dates.")
parser.add_argument(
    "--input_data",
    dest="input_data",
    help="Path to input dataset.",
)
parser.add_argument(
    "--et_hist_file",
    dest="et_hist_data",
    help="The historical 'Établissement' sirene database.",
)

parser.add_argument(
    "--output",
    dest="output",
    help="Path to the output dataset.",
)

args = parser.parse_args()

spark = sf_datalake.utils.get_spark_session()
df_et_hist = spark.read.csv(args.et_hist_data, header=True)
df_input = spark.read.csv(args.input_data, header=True, inferSchema=True)
df_et_hist = df_et_hist.select(
    [
        "siret",
        "etatAdministratifEtablissement",
        "dateDebut",
        "dateFin",
    ]
)
# dropna
df_et_hist = df_et_hist.dropna(subset=["dateDebut", "etatAdministratifEtablissement"])
# parse date
df_et_hist = df_et_hist.withColumn("dateDebut", F.to_date("dateDebut", "yyyy-mm-dd"))
df_et_hist = df_et_hist.withColumn("dateFin", F.to_date("dateFin", "yyyy-mm-dd"))
df_input = df_input.withColumn("periode", F.to_date("periode", "yyyy-mm-dd"))

# keep only active companies
df_et_hist = df_et_hist.filter(df_et_hist.etatAdministratifEtablissement == "A")
df_et_hist = df_et_hist.drop("etatAdministratifEtablissement")
df_et_hist = df_et_hist.withColumn(
    "dateFin",
    F.when(
        F.col("dateFin").isNull(), F.to_date(F.lit("2100-01-01"), "yyyy-MM-dd")
    ).otherwise(F.col("dateFin")),
)

output_ds = (
    df_et_hist.join(df_input, on=[df_et_hist.siret == df_input.siret,
                                df_input.periode >= df_et_hist.dateDebut, 
                                df_input.periode < df_et_hist.dateFin ] , 
                                how="inner")
).drop(df_et_hist.siret)

# Filter to keep only active periods

output_ds.write.options(header='True').csv(args.output)
