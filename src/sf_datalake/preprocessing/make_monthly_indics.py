"""Build monthly TVA data.

This follows MRV's process, originally written in SAS: `21_indicateurs.sas`
"""

from os import path

import pyspark.sql.functions as F  # pylint: disable=E0401
from pyspark.sql.types import StringType  # pylint: disable=E0401
from pyspark.sql.window import Window  # pylint: disable=E0401

from sf_datalake.preprocessing import DATA_ROOT_DIR, VUES_DIR
from sf_datalake.utils import load_source

####################
# Loading datasets #
####################

OUTPUT_FILE = path.join(DATA_ROOT_DIR, "tva.orc")

filename = {
    "t_art": "pub_risq_oracle.t_art.orc",
    "t_mvt": "pub_risq_oracle.t_mvt.orc",
    "t_mvr": "pub_risq_oracle.t_mvr.orc",
    "t_dar": "pub_risq_oracle.t_dar.orc",
    "t_dos": "pub_risq_oracle.t_dos.orc",
    "t_ech": "pub_risq_oracle.t_ech.orc",
    "af": "etl_decla-declarations_af.orc",  # pas la bonne table a priori
    "t_ref_etablissements": "pub_refent-t_ref_etablissements.orc",
    "t_ref_entreprise": "pub_refent-t_ref_entreprise.orc",
    "t_ref_code_nace": "pub_refer-t_ref_code_nace_complet.orc",
    "liasse_tva_ca3": "etl_tva.liasse_tva_ca3_view.orc",
    "t_etablissement_annee": "etl_refent-T_ETABLISSEMENT_ANNEE.orc",
}

t_art = load_source(path.join(VUES_DIR, filename["t_art"]))
t_mvt = load_source(path.join(VUES_DIR, filename["t_mvt"]))
t_dar = load_source(path.join(VUES_DIR, filename["t_dar"]))
t_dos = load_source(path.join(VUES_DIR, filename["t_dos"]))
t_mvr = load_source(path.join(VUES_DIR, filename["t_mvr"]))
t_ech = load_source(path.join(VUES_DIR, filename["t_ech"]))
t_ref_etablissements = load_source(
    path.join(VUES_DIR, filename["t_ref_etablissements"])
)
t_ref_entreprise = load_source(path.join(VUES_DIR, filename["t_ref_entreprise"]))
t_ref_code_nace = load_source(path.join(VUES_DIR, filename["t_ref_code_nace"]))
liasse_tva_ca3 = load_source(path.join(VUES_DIR, filename["liasse_tva_ca3"]))
t_etablissement_annee = load_source(
    path.join(VUES_DIR, filename["t_etablissement_annee"])
)

#######
# RAR #
#######

# Convert to dates
t_art = t_art.withColumn(
    "art_disc", F.to_date(F.col("art_disc").cast(StringType()), "yyyyMMdd")
)
t_art = t_art.withColumn(
    "art_didr", F.to_date(F.col("art_didr").cast(StringType()), "yyyyMMdd")
)
t_art = t_art.withColumn(
    "art_datedcf", F.to_date(F.col("art_datedcf").cast(StringType()), "yyyyMMdd")
)
t_art = t_art.withColumn(
    "art_dori", F.to_date(F.col("art_dori").cast(StringType()), "yyyyMMdd")
)
t_art = t_art.orderBy(["frp", "art_cleart"])  # useless on spark a priori ?

t_mvt = t_mvt.withColumn(
    "mvt_djc", F.to_date(F.col("mvt_djc").cast(StringType()), "yyyyMMdd")
)
t_mvt = t_mvt.withColumn(
    "mvt_deff", F.to_date(F.col("mvt_deff").cast(StringType()), "yyyyMMdd")
)
t_mvt = t_mvt.orderBy(["frp", "art_cleart"])  # useless on spark a priori ?

corresp_siren_frp2 = t_etablissement_annee.withColumn(
    "frp",
    F.concat(t_etablissement_annee.FRP_SERVICE, t_etablissement_annee.FRP_DOSSIER),
)

mvt_montant_creance = t_mvt.orderBy(["frp", "art_cleart"]).groupBy(
    ["frp", "art_cleart"]
)
mvt_montant_creance = mvt_montant_creance.sum("mvt_mdb").withColumnRenamed(
    "sum(mvt_mdb)", "mnt_creance"
)
mvt_montant_creance = t_mvt.join(
    mvt_montant_creance, on=["frp", "art_cleart"], how="left"
)

### Paiements
# Eventuellement faire un join car on perd des colonnes en faisant
# l'aggrégation. À voir.
mvt_paiement = t_mvt.filter("mvt_nacrd == 0 OR mvt_nacrd == 1")
mvt_paiement = mvt_paiement.withColumn(
    "mvt_djc_int", F.unix_timestamp(F.col("mvt_djc"))
)
mvt_paiement = mvt_paiement.orderBy("frp", "art_cleart", "mvt_djc").groupBy(
    ["frp", "art_cleart", "mvt_deff"]
)
mvt_paiement = mvt_paiement.agg(F.min("mvt_djc_int"), F.sum("mvt_mcrd"))
mvt_paiement = mvt_paiement.select(
    ["frp", "art_cleart", "min(mvt_djc_int)", "sum(mvt_mcrd)"]
)
mvt_paiement = mvt_paiement.withColumnRenamed("min(mvt_djc_int)", "min_mvt_djc_int")
mvt_paiement = mvt_paiement.withColumnRenamed("sum(mvt_mcrd)", "sum_mvt_mcrd")
mvt_paiement = mvt_paiement.dropDuplicates()

windowval = (
    Window.partitionBy("art_cleart")
    .orderBy(["frp", "min_mvt_djc_int"])
    .rangeBetween(Window.unboundedPreceding, 0)
)
mvt_paiement = mvt_paiement.filter("sum_mvt_mcrd != 0").withColumn(
    "mnt_paiement_cum", F.sum("sum_mvt_mcrd").over(windowval)
)
mvt_paiement = mvt_paiement.withColumn(
    "nb_paiement", F.count("sum_mvt_mcrd").over(windowval)
)
mvt_paiement = mvt_paiement.dropDuplicates()

# Paiements autres
mvt_paiement_autre = t_mvt.filter("mvt_nacrd != 0 AND mvt_nacrd != 1")
mvt_paiement_autre = mvt_paiement_autre.withColumn(
    "mvt_djc_int", F.unix_timestamp(F.col("mvt_djc"))
)
mvt_paiement_autre = mvt_paiement_autre.orderBy("frp", "art_cleart", "mvt_djc").groupBy(
    ["frp", "art_cleart", "mvt_deff"]
)
mvt_paiement_autre = mvt_paiement_autre.agg(F.min("mvt_djc_int"), F.sum("mvt_mcrd"))
mvt_paiement_autre = mvt_paiement_autre.withColumnRenamed(
    "min(mvt_djc_int)", "min_mvt_djc_int"
)
mvt_paiement_autre = mvt_paiement_autre.withColumnRenamed(
    "sum(mvt_mcrd)", "sum_mvt_mcrd"
)
mvt_paiement_autre = mvt_paiement_autre.dropDuplicates()

windowval = (
    Window.partitionBy("art_cleart")
    .orderBy(["frp", "min_mvt_djc_int"])
    .rangeBetween(Window.unboundedPreceding, 0)
)
mvt_paiement_autre = mvt_paiement_autre.filter("sum_mvt_mcrd != 0").withColumn(
    "mnt_paiement_cum_autre", F.sum("sum_mvt_mcrd").over(windowval)
)
mvt_paiement_autre = mvt_paiement_autre.withColumn(
    "nb_paiement_autre", F.count("sum_mvt_mcrd").over(windowval)
)
mvt_paiement_autre = mvt_paiement_autre.dropDuplicates()

# Join all tables
creances = t_art.join(mvt_montant_creance, on=["frp", "art_cleart"], how="left")
creances = creances.join(mvt_paiement, on=["frp", "art_cleart"], how="left")
creances = creances.join(mvt_paiement_autre, on=["frp", "art_cleart"], how="left")
creances = creances.join(corresp_siren_frp2, on=["frp"], how="left")

x_creances = creances.select(
    [
        "SIREN",
        "FRP",
        "ART_CLEART",
        "ART_NAART",
        "ART_NAPMC",
        "ART_DORI",
        "ART_DIDR",
        "ART_DATEDCF",
        "ART_DISC",
        "ART_ICIRREL",  # COM SAS : souvent pas rempli à ce niveau là
        "MNT_CREANCE",
        "MVT_DJC",
        "MNT_PAIEMENT_CUM",
        "MNT_PAIEMENT_CUM_AUTRE",
    ]
)

x_creances = x_creances.withColumn(
    "IND_CF", F.when(F.col("ART_DATEDCF").isNotNull(), 1).otherwise(0)
)
x_creances = x_creances.withColumn(
    "IND_HCF", F.when(F.col("ART_DATEDCF").isNotNull(), 0).otherwise(1)
)

# Je filtre pas sur les années (L254-L285 de 21_indicateurs.sas). On le fait à la
# modélisation ?

# TODO vérifier comment ces opérations se comportent vis à vis des nulls
rar_mois_article = x_creances.withColumn(
    "MNT_PAIEMENT_CUM_TOT",
    sum(x_creances[col] for col in ["MNT_PAIEMENT_CUM", "MNT_PAIEMENT_CUM_AUTRE"]),
)
rar_mois_article = rar_mois_article.withColumn(
    "MNT_PAIEMENT_CUM_TOT_HCF", F.col("MNT_PAIEMENT_CUM_TOT") * F.col("IND_HCF")
)
rar_mois_article = rar_mois_article.withColumn(
    "MNT_CREANCE_HCF", F.col("MNT_CREANCE") * F.col("IND_HCF")
)
rar_mois_article = rar_mois_article.withColumn(
    "MNT_RAR", F.col("MNT_CREANCE") - F.col("MNT_PAIEMENT_CUM_TOT")
)
rar_mois_article = rar_mois_article.withColumn(
    "MNT_RAR_HCF", F.col("MNT_RAR") * F.col("IND_HCF")
)

# Il manque des opérations qui n'ont plus de sens, il me semble, car on ne séléctionne
# plus les dates (L288-L306 de 21_indicateurs.sas)

# Write file
rar_mois_article.write.format("orc").save("/projets/TSF/sources/base/rar.orc")

# Les entreprises peuvent déclarer la TVA mensuellement, trimestriellement ou
# annuellement.
#
# On crée une variable pour distinguer les cas
ca3 = liasse_tva_ca3.withColumn(
    "duree_periode",
    F.round(
        F.months_between(
            liasse_tva_ca3.dte_fin_periode, liasse_tva_ca3.dte_debut_periode
        )
    ).cast("integer"),
)

x_tva = ca3.withColumn("D_TCA_TOTAL", F.col("D3310_29") + F.col("D3517S_55_i"))
x_tva = x_tva.withColumn(
    "D_TVA_NI_b0032_EXPORT", F.col("D3517S_02_b") + F.col("D3310_04")
)
x_tva = x_tva.withColumn("D_TVA_NI_b0034_LIC", F.col("D3517S_04_b") + F.col("D3310_06"))
x_tva = x_tva.withColumn(
    "D_TVA_NI_b0037_ACH_FRCH", F.col("D3517S_01_b") + F.col("D3310_07")
)
x_tva = x_tva.withColumn(
    "D_TVA_NI_b0029_LIV_EL_GAZ", F.col("D3517S_4D_b") + F.col("D3310_6A")
)
x_tva = x_tva.withColumn(
    "D_TVA_NI_b0043_ASSJT_HS_FR", F.col("D3517S_4B_b") + F.col("D3310_7A")
)
x_tva = x_tva.withColumn(
    "M_TVA_NI_b0033_AUTR_OP_NI",
    F.col("D3310_7B") + F.col("D3517S_03_b") + F.col("D3310_05"),
)
cols_SUM_TVA_NI_bTOTAL = [
    "D_TVA_NI_b0032_EXPORT",
    "D_TVA_NI_b0034_LIC",
    "D_TVA_NI_b0037_ACH_FRCH",
    "D_TVA_NI_b0029_LIV_EL_GAZ",
    "D_TVA_NI_b0043_ASSJT_HS_FR",
    "M_TVA_NI_b0033_AUTR_OP_NI",
]
x_tva = x_tva.withColumn(
    "SUM_TVA_NI_bTOTAL", sum(x_tva[col] for col in cols_SUM_TVA_NI_bTOTAL)
)
cols_M_TVA_BI_b0979_CA = [
    "D3310_01",  # "D3517S_05_b0206",
    "D3517S_5A_b",
    "D3517S_06_b",  # "D3517S_6B_b0150",
    "D3517S_6C_b",
    "D3517S_07_b",
    "D3517S_08_b",
    "D3517S_09_b",
    "D3517S_10_b",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0979_CA", sum(x_tva[col] for col in cols_M_TVA_BI_b0979_CA)
)
cols_M_TVA_BI_b0981_AUTR_OP_IMP = [
    "D3310_02",  # "D3310_2B",
    "D3310_3C",
    "D3517S_13_b",
    "D3517S_11_b",
    "D3517S_12_b",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0981_AUTR_OP_IMP",
    sum(x_tva[col] for col in cols_M_TVA_BI_b0981_AUTR_OP_IMP),
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_b0044_ACH_PS_IC", F.col("D3517S_AC_b") + F.col("D3310_2A")
)
x_tva = x_tva.withColumn("D_TVA_BI_b0031_AIC", F.col("D3517S_14_b") + F.col("D3310_03"))
x_tva = x_tva.withColumn(
    "D_TVA_BI_b0030_LIV_EL_GAZ", F.col("D3517S_AA_b") + F.col("D3310_3A")
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_b0040_ASSJT_HS_FR", F.col("D3517S_AB_b") + F.col("D3310_3B")
)
cols_SUM_TVA_BI_bTOTAL = [
    "M_TVA_BI_b0979_CA",
    "M_TVA_BI_b0981_AUTR_OP_IMP",
    "D_TVA_BI_b0044_ACH_PS_IC",
    "D_TVA_BI_b0031_AIC",
    "D_TVA_BI_b0030_LIV_EL_GAZ",
    "D_TVA_BI_b0040_ASSJT_HS_FR",
]
x_tva = x_tva.withColumn(
    "SUM_TVA_BI_bTOTAL", sum(x_tva[col] for col in cols_SUM_TVA_BI_bTOTAL)
)
x_tva = x_tva.withColumn(
    "SUM_TVA_NI_BI_bTOTAL", F.col("SUM_TVA_BI_bTOTAL") + F.col("SUM_TVA_NI_bTOTAL")
)
cols_M_TVA_BI_b0207_NORMAL = [  # "D3517S_05_b0206",
    "D3310_08_btx196",
    "D3517S_5A_b",
    "D3310_08_b",
    "D3517S_11_b",
    "D3517S_12_b",
    "D3517S_13_b",
    "D3517S_14_b",
    "D3517S_AB_b",
    "D3517S_AC_b",
    "D3517S_AA_b",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0207_NORMAL", sum(x_tva[col] for col in cols_M_TVA_BI_b0207_NORMAL)
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0105_REDUIT_5_5", F.col("D3517S_06_b") + F.col("D3310_09_b")
)
cols_M_TVA_BI_b0151_REDUIT_10 = [  # D3517S_6B_b0150,
    "D3310_9B_btx7",
    "D3517S_6C_b",
    "D3310_9B_b",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0151_REDUIT_10", sum(x_tva[col] for col in cols_M_TVA_BI_b0151_REDUIT_10)
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0100_DOM_2_1", F.col("D3517S_08_b") + F.col("D3310_11_b")
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0201_DOM_8_5", F.col("D3517S_07_b") + F.col("D3310_10_b")
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_b0950_TX_PART", F.col("D3517S_09_b") + F.col("D3310_14_b")
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_b0900_ANC_TX", F.col("D3517S_10_b") + F.col("D3310_13_b")
)
x_tva = x_tva.withColumn(
    "D_TVA_COL_i0600_ANT_DED", F.col("D3310_15") + F.col("D3517S_18_i")
)
x_tva = x_tva.withColumn(
    "SUM_TVA_COL_TOTAL",
    F.col("D3310_16")
    - F.col("D3310_15")
    + F.col("D3517S_16_i")
    - F.col("D3310_7C")
    - F.col("D3310_17")
    - F.col("D3310_5B")
    - F.col("D3517S_AA_i")
    - F.col("D3517S_AB_i")
    - F.col("D3517S_AC_i")
    - F.col("D3517S_13_i")
    - F.col("D3517S_14_i"),
)
x_tva = x_tva.withColumn(
    "D_TVA_COL_i0031_AIC", F.col("D3517S_14_i") + F.col("D3310_17")
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_i0703_IMM", F.col("D3310_19") + F.col("D3517S_23_i")
)
x_tva = x_tva.withColumn(
    "M_TVA_DED_i0702_ABS",
    F.col("D3310_20") + F.col("D3517S_20_i") + F.col("D3517S_21_i"),
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_i0059_AUTR", F.col("D3310_21") + F.col("D3517S_25_i")
)  # D3310_2C
# TODO if (D3310_22A=. AND D3517S_25A_TX_DED=.) then D_TVA_DED_TX_COEF_DED = 100; else
# D_TVA_DED_TX_COEF_DED = SUM(D3310_22A,D3517S_25A_TX_DED) ;
x_tva = x_tva.withColumn(
    "D_TVA_DED_i0705_TOTAL", F.col("D3310_23") + F.col("D3517S_26_i")
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_TOTAL_HS_REPORT",
    F.col("D_TVA_DED_i0703_IMM")
    + F.col("M_TVA_DED_i0702_ABS")
    + F.col("D_TVA_DED_i0059_AUTR"),
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_i0709_DT_ES_DOM", F.col("D3310_24") + F.col("D3517S_27_i")
)
x_tva = x_tva.withColumn(
    "M_TVA_NET_i8002_REMB_DEM", F.col("D3310_26") + F.col("D3517S_50_i")
)
x_tva = x_tva.withColumn("M_TVA_NET_DUE", F.col("D3310_28") + F.col("D3517S_28_i"))

# Write file
x_tva.write.format("orc").save(OUTPUT_FILE)
