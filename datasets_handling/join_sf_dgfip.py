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
# data_paths = {
#     "sf": args.sf_data,
#     "yearly": path.join(args.dgfip_dir, ""),
#     "monthly": path.join(args.dgfip_dir, ""),
# }

## Load datasets
datasets = load_data({"sf": args.sf_data, "dgfip": args.dgfip_data}, file_format="orc")

## Join datasets on SIREN and monthly dates.
df_dgfip = sf_datalake.transform.stringify_and_pad_siren(datasets["dgfip"])
df_dgfip = sf_datalake.transform.explode_between_dates(
    df_dgfip, "date_deb_exercice", "date_fin_exercice"
)

### synchronizing features in time
df_sf = sf_datalake.transform.stringify_and_pad_siren(datasets["sf"]).withColumn(
    "periode",
    F.date_trunc("month", F.to_date(F.col("periode"))),
)

df_joined = df_sf.join(df_dgfip, on=["periode", "siren"], how="left")

df_joined.write.format("orc").save(args.output)
