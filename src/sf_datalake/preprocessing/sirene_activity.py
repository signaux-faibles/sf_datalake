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


parser = argparse.ArgumentParser(
    description="Filter company on activity date"
)
parser.add_argument(
    "--sf",
    dest="sf_data",
    help="Path to the Signaux Faibles dataset.",
)
parser.add_argument(
    "--et_hist_file",
    dest="et_hist_data",
    help="The historical 'Établissement' database.",
)

parser.add_argument(
    "--output",
    dest="output",
    help="Path to the output dataset.",
)

args = parser.parse_args()

spark = sf_datalake.utils.get_spark_session()
#Load data
df_et_hist = spark.read.csv(args.et_hist_data, header=True)
df_et_hist = df_et_hist.select(
    ["siret",
        "etatAdministratifEtablissement",
        "dateDebut",
        "dateFin",                                   
])
# dropna
df_et_hist = df_et_hist.dropna(subset=["dateDebut", "etatAdministratifEtablissement"])
# parse date
df_et_hist = df_et_hist.withColumn("dateDebut", F.to_date("dateDebut","yyyy-mm-dd"))
df_et_hist = df_et_hist.withColumn("dateFin", F.to_date("dateFin","yyyy-mm-dd"))
# keep only head office
df_et_hist = df_et_hist.filter(df_et_hist.etatAdministratifEtablissement == "A")
df_et_hist = df_et_hist.drop("etatAdministratifEtablissement")

df_sf = spark.read.csv(args.sf_data, header=True)

output_ds = (
    df_et_hist.join(
        df_sf,
        on = (df_et_hist.siret == df_sf.siret),
        how = "inner")
    ).drop(df_et_hist.siret)

output_ds.write.format("orc").save(args.output)
