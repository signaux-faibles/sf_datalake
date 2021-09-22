from os import path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

#  Instanciating Spark session

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.shuffle.blockTransferService", "nio")
spark.conf.set("spark.driver.maxResultSize", "1300M")

#########
# Utils #
#########


def load_source(src_name, spl_size=None):
    df = spark.read.orc(SRC_PATHS[src_name])
    if spl_size is not None:
        df = df.sample(spl_size)
    return df


def null_values_stats(df, columns=None):
    """Computes percentage of null values in given DataFrame columns."""
    if columns is None:
        columns = df.columns
    df_size = df.count()
    return df.select(
        [(F.count(F.when(F.isnull(c), c)) / df_size).alias(c) for c in columns]
    )


####################
# Loading datasets #
####################

# SAMPLE_SIZE = 0.1  # set to None to get all dataset
SAMPLE_SIZE = None

ROOT_FOLDER = "/projets/TSF/sources/"
MRV_DTNUM_FOLDER = path.join(ROOT_FOLDER, "livraison_MRV-DTNUM_juin_2021")

SRC_PATHS = {
    "decla_indmap": path.join(MRV_DTNUM_FOLDER, "etl_decla-declarations_indmap.orc"),
    "decla_af": path.join(MRV_DTNUM_FOLDER, "etl_decla-declarations_af.orc"),
    "defaillances": path.join(MRV_DTNUM_FOLDER, "pub_medoc_oracle-t_defaillance.orc"),
    "refent_etab": path.join(MRV_DTNUM_FOLDER, "pub_refent-t_ref_etablissements.orc"),
    "refent_entr": path.join(MRV_DTNUM_FOLDER, "pub_refent-t_ref_entreprise.orc"),
    "jugements": path.join(MRV_DTNUM_FOLDER, "etl_refent_oracle-t_jugement_histo.orc"),
    "rar_tva": path.join(MRV_DTNUM_FOLDER, "rar.rar_tva_exercice.orc"),
    "sf": path.join(ROOT_FOLDER, "data_sf_padded.orc"),
}

indmap = load_source("decla_indmap", SAMPLE_SIZE)
af = load_source("decla_af", SAMPLE_SIZE)
defa = load_source("defaillances", SAMPLE_SIZE)
refent_entr = load_source("refent_entr", SAMPLE_SIZE)
refent_etab = load_source("refent_etab", SAMPLE_SIZE)
jugements = load_source("jugements", SAMPLE_SIZE)
rar_tva = load_source("rar_tva", SAMPLE_SIZE)

sf = load_source("sf", SAMPLE_SIZE)

####################
# Merge datasets   #
####################

# Building yearly indicators, following MRV's model (originally in SAS)

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

## Jointure taux d'accroissement

df_v = df.join(
    tac,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

## Jointure SF

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

indics_annuels.write.format("orc").save(
    "/projets/TSF/sources/base/indicateurs_annuels.orc"
)
