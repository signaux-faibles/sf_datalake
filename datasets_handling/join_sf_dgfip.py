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
from sf_datalake.io import load_data
from sf_datalake.transform import stringify_and_pad_siren

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
parser.add_argument(
    "--diff",
    dest="bilan_periode_diff",
    help="""Difference between 'arrete_bilan_diane' and 'periode' that will be
    used to complete missing diane accounting year end date (used as a join key).
    """,
    type=int,
    default=-392,
)

args = parser.parse_args()
# data_paths = {
#     "sf": args.sf_data,
#     "yearly": path.join(args.dgfip_dir, ""),
#     "monthly": path.join(args.dgfip_dir, ""),
# }

## Load datasets

datasets = load_data({"sf": args.sf_data, "dgfip": args.dgfip_data}, file_format="orc")

## Join datasets

df_dgfip = datasets["dgfip"].withColumn(
    "join_date", F.year(datasets["dgfip"]["date_fin_exercice"])
)
df_sf = stringify_and_pad_siren(datasets["sf"]).withColumn(
    "join_date",
    F.year(
        F.coalesce(
            F.col("arrete_bilan_diane"),
            F.last_day(F.date_add(F.col("periode"), args.bilan_periode_diff)),
        )
    ),
)

df_joined = df_sf.join(
    df_dgfip,
    on=[
        "join_date",
        "siren",
    ],
    how="full_outer",
)

df_joined.write.format("orc").save(path.join(args.output))
