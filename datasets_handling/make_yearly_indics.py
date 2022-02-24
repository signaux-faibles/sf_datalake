"""Build a dataset of yearly DGFiP data.

This follows MRV's process, originally written in SAS.

USAGE
    python make_yearly_data.py <DGFiP_vues_directory> <output_directory>
"""
from os import path

import pyspark.sql.functions as F
from pyspark.sql.window import Window

import sf_datalake.io
from sf_datalake import DGFIP_VARIABLES

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser("orc")
parser.description = "Build a dataset of yearly DGFiP data."
args = parser.parse_args()

data_paths = {
    "indmap": path.join(args.input_dir, "etl_decla-declarations_indmap"),
    "af": path.join(args.input_dir, "etl_decla-declarations_af"),
    "rar_tva": path.join(args.input_dir, "rar.rar_tva_exercice"),
}
datasets = sf_datalake.io.load_data(data_paths)

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

# Jointure RAR_TVA
df = df.join(
    datasets["rar_tva"],
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

# Calcul taux d'accroissement
df = df.withColumn(
    "per_rank",
    F.dense_rank().over(Window.partitionBy("siren").orderBy("date_deb_exercice")),
).drop_duplicates(
    subset=["siren", "per_rank"]
)  # 2 obs with the same "date_deb_exercice" --> only keep 1

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

##  'taux d'accroissement' DataFrame join

indics_annuels = df.join(
    tac,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)


indics_annuels.write.format("orc").save(args.output_dir)
