from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as f
from pyspark.sql import SparkSession, types
from pyspark.ml.feature import VectorAssembler

## Instanciating Spark session
spark = SparkSession.builder.getOrCreate()

#########
# Utils #
#########

def load_source(src_name, spl_size = None):
    df = spark.read.orc(f"{vues_rootfolder}/{src[src_name]}")
    if spl_size is not None:
        df = df.sample(n=spl_size)
    return df

####################
# Loading datasets #
####################

vues_rootfolder = "/projets/TSF/sources/livraison_MRV-DTNUM_juin_2021"

src = {
    "decla_indmap":  "etl_decla-declarations_indmap.orc",
    "decla_af": "etl_decla-declarations_af.orc",
    "defaillances": "pub_medoc_oracle-t_defaillance.orc",
    "refent_etab": "pub_refent-t_ref_etablissements.orc",
    "refent_entr": "pub_refent-t_ref_entreprise.orc",
    "jugements": "etl_refent_oracle-t_jugement_histo.orc",
    "signaux_faibles": "export_clean.orc",
    "groupes": "",   
}

indmap = load_source("decla_indmap")
af = load_source("decla_af")
defa = load_source("defaillances")
refent_entr = load_source("refent_entr")
refent_etab = load_source("refent_etab")
jugements = load_source("jugements")

sf = spark.read.options(inferSchema='True', header="True", sep=",").csv("/projets/TSF/sources/data_sf.csv")

#####
# Building yearly indicators, following MRV's model (originally in SAS)
####

df = indmap.join(
    af,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left"
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
).withColumn(
    "year_exercice", f.year("date_fin_exercice")
)


from pyspark.sql.window import Window

df = df.withColumn(
    "per_rank", f.dense_rank().over(
        Window.partitionBy("siren").orderBy("date_deb_exercice")
    )
)

df_ante = df.alias("df_ante")
for col in df_ante.columns:
    df_ante = df_ante.withColumnRenamed(
        col,
        f"{col}_ante"
    )

tac_base = df.join(
    df_ante,
    on=[
        df_ante.siren_ante == df.siren,
        df_ante.per_rank_ante+2 == df.per_rank,
    ],
    how="left"
)

tac_columns = []
key_columns = ["siren", "date_deb_exercice", "date_fin_exercice"]
skip_columns = ["year_exercice", "per_rank"]

for col in df.columns:
    if col in (key_columns + skip_columns):
        continue
    tac_base = tac_base.withColumn(
        f"tac_1y_{col}",
        (tac_base[col]-tac_base[f"{col}_ante"])/(tac_base[f"{col}_ante"])
    )
    tac_columns.append(f"tac_1y_{col}")
    
tac = tac_base.select(tac_columns+key_columns)

df_v = df.join(
    tac,
    on=["siren", "date_deb_exercice", "date_fin_exercice"],
    how="left",
)

indics_annuels = df_v.join(
    sf.withColumnRenamed("siren", "siren_sf"),
    [
        f.months_between(
            f.to_date(sf["periode"]),
            f.to_date(df_v["date_deb_exercice"]),
        ) >= 0,
        f.months_between(
            f.to_date(sf["periode"]),
            f.to_date(df_v["date_fin_exercice"]),
        ) <= 0,
    ],
    how = "full"
)

indics_annuels = indics_annuels.filter(indics_annuels.siren == indics_annuels.siren_sf).drop("siren_sf")

indics_annuels.write.format("orc").save("/projets/TSF/sources/base/indicateurs_annuels.orc")
