"""Build yearly indicators data.

This follows MRV's process, originally written in SAS.
"""

from os import path

import pyspark.sql.functions as F
from pyspark.sql.window import Window

from sf_datalake.preprocessing import DATA_ROOT_DIR, VUES_DIR
from sf_datalake.utils import load_source

####################
# Loading datasets #
####################

filename = {
    "decla_indmap": "etl_decla-declarations_indmap.orc",
    "decla_af": "etl_decla-declarations_af.orc",
    "defaillances": "pub_medoc_oracle-t_defaillance.orc",
    "refent_etab": "pub_refent-t_ref_etablissements.orc",
    "refent_entr": "pub_refent-t_ref_entreprise.orc",
    "jugements": "etl_refent_oracle-t_jugement_histo.orc",
    "rar_tva": "rar.rar_tva_exercice.orc",
    "sf": "data_sf_padded.orc",
}

OUTPUT_PATH = path.join(DATA_ROOT_DIR, "/base/indicateurs_annuels.orc")

indmap = load_source(path.join(VUES_DIR, filename["decla_indmap"]))
af = load_source(path.join(VUES_DIR, filename["decla_af"]))
defa = load_source(path.join(VUES_DIR, filename["defaillances"]))
refent_entr = load_source(path.join(VUES_DIR, filename["refent_entr"]))
refent_etab = load_source(path.join(VUES_DIR, filename["refent_etab"]))
jugements = load_source(path.join(VUES_DIR, filename["jugements"]))
rar_tva = load_source(path.join(VUES_DIR, filename["rar_tva"]))

sf = load_source(path.join(DATA_ROOT_DIR, filename["sf"]))

####################
# Merge datasets   #
####################

df = indmap.join(
    af, on=["siren", "date_deb_exercice", "date_fin_exercice"], how="left"
).select(
    "siren",
    "date_deb_exercice",
    "date_fin_exercice",
    "MNT_AF_CA",
    "MNT_AF_SIG_EBE",
    "RTO_AF_SOLV_ENDT_NET",
    "MNT_AF_BPAT_PASSIF_K_PROPRES",
    "MNT_AF_SIG_RCAI",
    "MNT_AF_BFONC_TRESORERIE",
    "RTO_AF_AUTO_FINANCIERE",
    "RTO_AF_SOLV_INDP_FI",
    "MNT_AF_BFONC_ACTIF_TRESORERIE",
    "MNT_AF_BFONC_PASSIF_TRESORERIE",
    "MNT_AF_BPAT_ACTIF_SUP1AN",
    "MNT_AF_BPAT_ACTIF_STOCKS",
    "MNT_AF_BPAT_ACTIF_DISPO",
    "RTO_AF_SOLV_ENDT_BRUT",
    "MNT_AF_BFONC_RESSOUR_STABL",
    "MNT_AF_BFONC_FRNG",
    "MNT_AF_BPAT_ACTIF_CREANCES",
    "RTO_AF_SOLV_SOLVABILITE",
    "RTO_AF_SOLV_LQDT_RESTRINTE",
    "RTO_AF_STRUCT_FIN_IMM",
    "RTO_AF_SOLIDITE_FINANCIERE",
    "TX_MOY_TVA_COL",
    "NBR_JOUR_RGLT_CLI",
    "NBR_JOUR_RGLT_FRS",
    "TOT_CREA_CHAV_MOINS1AN",
    "TOT_DET_PDTAV_MOINS1AN",
    "PCT_AF_SOLV_LQDT_GEN",
    "PCT_REND_EXPL",
    "RTO_MG_ACHAT_REV",
    "PCT_CHARG_EXTE_CA_NET",
    "PCT_INDEP_FIN",
    "PCT_PDS_INTERET",
    "NBR_JOUR_ROTA_STK",
    "RTO_INVEST_CA",
    "RTO_TVA_COL_FR",
    "D_CMPT_COUR_ASSO_DEB",
    "D_CMPT_COUR_ASSO_CRED",
    "RTO_TVA_DEDUC_ACH",
    "RTO_TVA_DECUC_TVA_COL",
    "D_CR_250_EXPL_SALAIRE",
    "D_CR_252_EXPL_CH_SOC",
    "MNT_AF_SIG_EBE_RET",
    "MNT_AF_BFONC_BFR",
    "RTO_AF_RATIO_RENT_MBE",
    "RTO_AF_RENT_ECO",
)

# Jointure RAR_TVA
df = df.join(
    rar_tva, on=["siren", "date_deb_exercice", "date_fin_exercice"], how="left"
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
    if col in key_columns + skip_columns:
        continue
    tac_base = tac_base.withColumn(
        f"tac_1y_{col}",
        (tac_base[col] - tac_base[f"{col}_ante"]) / (tac_base[f"{col}_ante"]),
    )
    tac_columns.append(f"tac_1y_{col}")

tac = tac_base.select(tac_columns + key_columns)

##  'taux d'accroissement' DataFrame join

df_v = df.join(
    tac,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

## SF join

df_v = df_v.withColumn(
    "year_dgfip", F.year(df_v["date_fin_exercice"])
).withColumnRenamed("siren", "siren_dgfip")

sf = sf.withColumn(
    "year",
    F.when(sf["arrete_bilan_bdf"].isNotNull(), F.year(sf["arrete_bilan_bdf"]))
    .when(
        (sf["exercice_diane"].isNotNull()) & (sf["arrete_bilan_bdf"].isNull()),
        sf["exercice_diane"],
    )
    .otherwise(F.year(sf["periode"])),
).withColumn("siren", F.substring(sf.siret, 1, 9))

indics_annuels = sf.join(
    df_v,
    on=[sf.year == df_v.year_dgfip, sf.siren == df_v.siren_dgfip],
    how="full_outer",
)

indics_annuels.write.format("orc").save(OUTPUT_PATH)
