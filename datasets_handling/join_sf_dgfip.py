"""Build a dataset by joining DGFiP and Signaux Faibles data.

The join is made along temporal and SIREN variables.

USAGE
    python join_sf_dgfip.py --sf <sf_dataset> --dgfip <DGFiP_dataset> \
    --output <output_directory>

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
import sf_datalake.transform
from sf_datalake.io import load_data

parser = argparse.ArgumentParser(
    description="Merge DGFiP and Signaux Faibles datasets into a single one."
)
parser.add_argument(
    "--sf",
    dest="sf_data",
    help="Path to the Signaux Faibles dataset.",
)
parser.add_argument(
    "--dgfip",
    dest="dgfip_data",
    help="Path to the DGFiP dataset.",
)
parser.add_argument(
    "--output",
    dest="output",
    help="Path to the output dataset.",
)

args = parser.parse_args()

# Load datasets
datasets = load_data({"sf": args.sf_data, "dgfip": args.dgfip_data}, file_format="orc")

df_dgfip = sf_datalake.transform.stringify_and_pad_siren(datasets["dgfip"])
df_sf = sf_datalake.transform.stringify_and_pad_siren(datasets["sf"]).withColumn(
    "periode", F.to_date(F.date_trunc("month", F.col("periode")))
)

# Normalize accounting year by duration
for c in sf_datalake.utils.numerical_columns(df_dgfip):
    if c != "duree_exercice":
        df_dgfip = df_dgfip.withColumn(c, F.col(c) / F.col("duree_exercice"))

# Hack to consider time periods that span whole months from beginning to end.
df_dgfip = df_dgfip.withColumn(
    "date_fin_exercice",
    F.add_months(F.to_date(F.date_trunc("month", F.col("date_fin_exercice"))), 1),
).withColumn(
    "date_deb_exercice",
    F.to_date(F.date_trunc("month", F.col("date_deb_exercice"))),
)

# Join datasets and drop (time, SIREN) duplicates with the highest null values ratio
df_dgfip = df_dgfip.withColumn(
    "null_ratio",
    sum([F.when(F.col(c).isNull(), 1).otherwise(0) for c in df_dgfip.columns])
    / len(df_dgfip.columns),
)

df_joined = (
    df_sf.join(
        df_dgfip,
        on=(
            (df_sf.siren == df_dgfip.siren)
            & (df_sf.periode >= df_dgfip.date_deb_exercice)
            & (df_sf.periode < df_dgfip.date_fin_exercice)
        ),
        how="left",
    )
    .drop(df_dgfip.siren)
    .orderBy("siren", "periode", "null_ratio")
    .dropDuplicates(["siren", "periode"])
    .drop("null_ratio")
)

df_joined.write.format("orc").save(args.output)
