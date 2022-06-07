"""Build a dataset by joining DGFiP and Signaux Faibles data.

The join is made along temporal and SIREN variables.

USAGE
    python join_datasets.py --sf <sf_dataset> --dgfip_yearly <DGFiP_yearly_dataset> \
    --dgfip_tva <DGFiP_TVA_dataset> --dgfip_rar <DGFiP_rar_dataset> \
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
    "--dgfip_yearly",
    help="Path to the DGFiP yearly dataset.",
)
parser.add_argument(
    "--dgfip_tva",
    help="Path to the DGFiP tva dataset.",
)
parser.add_argument(
    "--dgfip_rar",
    help="Path to the DGFiP rar 'reste à recouvrer' monthly dataset.",
)
parser.add_argument(
    "--output",
    dest="output",
    help="Path to the output dataset.",
)

args = parser.parse_args()

# Load datasets
datasets = load_data(
    {
        "sf": args.sf_data,
        "dgfip_yearly": args.dgfip_yearly,
        "dgfip_tva": args.dgfip_tva,
        "dgfip_rar": args.dgfip_rar,
    },
    file_format="orc",
)


# Prepare datasets
df_dgfip_yearly = sf_datalake.transform.stringify_and_pad_siren(
    datasets["dgfip_yearly"]
)
df_dgfip_tva = sf_datalake.transform.stringify_and_pad_siren(datasets["dgfip_tva"])
df_dgfip_rar = sf_datalake.transform.stringify_and_pad_siren(datasets["dgfip_rar"])
df_sf = sf_datalake.transform.stringify_and_pad_siren(datasets["sf"]).withColumn(
    "periode", F.to_date(F.date_trunc("month", F.col("periode")))
)
df_dgfip_yearly = df_dgfip_yearly.withColumn(
    "null_ratio",
    sum([F.when(F.col(c).isNull(), 1).otherwise(0) for c in df_dgfip_yearly.columns])
    / len(df_dgfip_yearly.columns),
)
df_dgfip_rar = df_dgfip_rar.filter(
    F.col("siren").isNotNull()
    & F.col("art_didr").isNotNull()
    & F.col("mvt_djc").isNotNull()
    & F.col("mnt_creance").isNotNull()
)

# Join datasets and drop (time, SIREN) duplicates with the highest null values ratio
overwritten_columns = (set(df_dgfip_yearly.columns) & set(df_dgfip_tva.columns)) - {
    "siren"
}
joined_df = (
    df_sf.drop(*overwritten_columns)
    .join(
        df_dgfip_yearly,
        on=(
            (df_sf.siren == df_dgfip_yearly.siren)
            & (df_sf.periode >= df_dgfip_yearly.date_deb_exercice)
            & (df_sf.periode < df_dgfip_yearly.date_fin_exercice)
        ),
        how="left",
    )
    .drop(df_dgfip_yearly.siren)
    .orderBy("null_ratio")
    .dropDuplicates(["siren", "periode"])
    .drop("null_ratio")
)

joined_df = joined_df.join(
    df_dgfip_tva,
    on=(
        (joined_df.siren == df_dgfip_tva.siren)
        & (joined_df.periode >= df_dgfip_tva.date_deb_tva)
        & (joined_df.periode < df_dgfip_tva.date_fin_tva)
    ),
    how="left",
).drop(df_dgfip_tva.siren)

joined_df = (
    joined_df.join(
        df_dgfip_rar,
        on=(
            (joined_df.siren == df_dgfip_rar.siren)
            & (joined_df.periode >= df_dgfip_rar.mvt_djc)
            & (joined_df.periode >= df_dgfip_rar.art_didr)
        ),
        how="left",
    )
    .sort(["mvt_djc", "mnt_paiement_cum_tot"], ascending=False)
    .dropDuplicates(["siren", "periode", "art_cleart"])
).drop(df_dgfip_rar.siren)

joined_df.write.format("orc").save(args.output)
