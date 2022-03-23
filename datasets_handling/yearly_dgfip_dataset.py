"""Build a dataset of yearly DGFiP data.

This follows MRV's process, originally written in SAS. Source data should be stored
beforehand inside an input directory which, in turn, contains 3 directories containing
the data as (possibly multiple) orc file(s):
- etl_decla-declarations_indmap
- etl_decla-declarations_af
- rar.rar_tva_exercice

A yearly dataset will be stored as split orc files under the chosen output directory.

USAGE
    python make_yearly_data.py <DGFiP_tables_directory> <output_directory>

"""
import os
import sys
from os import path

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.io
from sf_datalake import DGFIP_VARIABLES

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset of yearly DGFiP data."
args = parser.parse_args()

data_paths = {
    "indmap": path.join(args.input, "etl_decla-declarations_indmap"),
    "af": path.join(args.input, "etl_decla-declarations_af"),
    "rar_tva": path.join(args.input, "rar.rar_tva_exercice"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="orc")

###################
# Merge datasets  #
###################

df = (
    datasets["indmap"]
    .join(
        datasets["af"],
        on=["siren", "date_deb_exercice", "date_fin_exercice"],
        how="left",
    )
    .select(DGFIP_VARIABLES)
)

# Join RAR_TVA
df = df.join(
    datasets["rar_tva"],
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

# Drop duplicates
df = df.withColumn(
    "per_rank",
    F.dense_rank().over(Window.partitionBy("siren").orderBy("date_deb_exercice")),
).drop_duplicates(subset=["siren", "per_rank"])
# TODO: review this drop. Here, 2 obs with the same "date_deb_exercice" --> only keep 1

# Compute variations
df_ante = df.alias("df_ante")
for col in df_ante.columns:
    df_ante = df_ante.withColumnRenamed(col, f"{col}_ante")

tac_base = df.join(
    df_ante,
    on=[
        df_ante.siren_ante == df.siren,
        df_ante.per_rank_ante + 2 == df.per_rank,
    ],
    how="left",
)

tac_columns = []
key_columns = ["siren", "date_deb_exercice", "date_fin_exercice"]
skip_columns = ["per_rank"]

for col in df.columns:
    if not col in key_columns + skip_columns:
        tac_base = tac_base.withColumn(
            f"tac_1y_{col}",
            (tac_base[col] - tac_base[f"{col}_ante"]) / (tac_base[f"{col}_ante"]),
        )
        tac_columns.append(f"tac_1y_{col}")

tac = tac_base.select(tac_columns + key_columns)

# 'taux d'accroissement' DataFrame join
indics_annuels = df.join(
    tac,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

indics_annuels.write.format("orc").save(args.output)
