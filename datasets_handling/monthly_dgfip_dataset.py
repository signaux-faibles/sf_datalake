"""Build a dataset of monthly TVA data.

This follows MRV's process, originally written in SAS: `21_indicateurs.sas`. Source data
should be stored beforehand inside an input directory which, in turn, contains the 4
following directories containing the data as (possibly multiple) orc file(s):
- t_art (pub_risq_oracle)
- t_mvt (pub_risq_oracle)
- liasse_tva_ca3_view (etl_tva)
- etl_refent-T_ETABLISSEMENT_ANNEE

USAGE
    python make_monthly_data.py <DGFiP_vues_directory> <output_directory>

"""

import os
import sys
from os import path

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable = C0413
import sf_datalake.io
import sf_datalake.transform

## Utility function


def process_payment(
    df: pyspark.sql.DataFrame, suffix: str = None
) -> pyspark.sql.DataFrame:
    """Computes the number and amounts of payments.

    Args:
        df: A DataFrame containing payment data.
        suffix: A suffix for output column names.

    Returns:
        A DataFrame with two new columns "nb_paiement" and "mnt_paiement_cum".

    """
    output_cols = {"mnt": "MNT_PAIEMENT_CUM", "nb": "NB_PAIEMENT"}
    if suffix is not None:
        output_cols = {key: f"{col}_{suffix}" for key, col in output_cols.items()}
    windowval = (
        Window.partitionBy("ART_CLEART")
        .orderBy(["FRP", "min(MVT_DJC_INT)"])
        .rangeBetween(Window.unboundedPreceding, 0)
    )

    df = df.withColumn("MVT_DJC_INT", F.unix_timestamp(F.col("MVT_DJC")))
    gb = df.orderBy("FRP", "ART_CLEART", "MVT_DJC").groupBy(
        ["FRP", "ART_CLEART", "MVT_DEFF"]
    )
    df_agg = (
        gb.agg(F.min("MVT_DJC_INT"), F.sum("MVT_MCRD"))
        .select(["FRP", "ART_CLEART", "min(MVT_DJC_INT)", "sum(MVT_MCRD)"])
        .dropDuplicates()
    )
    return (
        df_agg.filter(F.col("sum(MVT_MCRD)") != 0)
        .withColumn(
            output_cols["mnt"],
            F.sum("sum(MVT_MCRD)").over(windowval),
        )
        .withColumn(output_cols["nb"], F.count("sum(MVT_MCRD)").over(windowval))
        .dropDuplicates()
        .drop(*["sum(MVT_MCRD)", "min(MVT_DJC_INT)"])
    )


####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset of monthly/quarterly TVA data."
args = parser.parse_args()

data_paths = {
    "t_art": path.join(args.input, "t_art"),
    "t_mvt": path.join(args.input, "t_mvt"),
    "liasse_tva_ca3": path.join(args.input, "liasse_tva_ca3_view"),
    "liasse_tva_ca12": path.join(args.input, "liasse_tva_ca12_view"),
    "t_etablissement_annee": path.join(args.input, "etl_refent-T_ETABLISSEMENT_ANNEE"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="orc")

# Set every column name to upper case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.upper() for col in ds.columns))


#######
# RAR #
#######

# Parse dates, create "frp" join key

t_art = sf_datalake.transform.parse_date(
    datasets["t_art"], ["ART_DISC", "ART_DIDR", "ART_DATEDCF", "ART_DORI"]
)
t_mvt = sf_datalake.transform.parse_date(datasets["t_mvt"], ["MVT_DJC", "MVT_DEFF"])

corresp_siren_frp2 = (
    datasets["t_etablissement_annee"]
    .withColumn(
        "FRP",
        F.concat(
            datasets["t_etablissement_annee"].FRP_SERVICE,
            datasets["t_etablissement_annee"].FRP_DOSSIER,
        ),
    )
    .drop(*["FRP_SERVICE", "FRP_DOSSIER"])
)

# Eventuellement faire un join car on perd des colonnes en faisant
# l'aggrégation. À voir.

mvt_montant_creance = t_mvt.join(
    t_mvt.groupBy(["FRP", "ART_CLEART"])
    .sum("MVT_MDB")
    .withColumnRenamed("sum(MVT_MDB)", "MNT_CREANCE"),
    on=["FRP", "ART_CLEART"],
    how="left",
).drop(*["ID", "DATE_CHARGEMENT", "FRP_SERVICE", "FRP_DOSSIER"])


mvt_paiement_nacrd01 = process_payment(t_mvt.filter("MVT_NACRD == 0 OR MVT_NACRD == 1"))
mvt_paiement_nacrd_autre = process_payment(
    t_mvt.filter("MVT_NACRD != 0 AND MVT_NACRD != 1"), suffix="AUTRE"
)

# Join all tables
creances = (
    t_art.join(mvt_montant_creance, on=["FRP", "ART_CLEART"], how="left")
    .join(
        mvt_paiement_nacrd01.drop(*["FRP_SERVICE", "FRP_DOSSIER"]),
        on=["FRP", "ART_CLEART"],
        how="left",
    )
    .join(
        mvt_paiement_nacrd_autre.drop(*["FRP_SERVICE", "FRP_DOSSIER"]),
        on=["FRP", "ART_CLEART"],
        how="left",
    )
    .join(corresp_siren_frp2, on=["FRP"], how="left")
)

x_creances = creances.select(
    [
        "SIREN",
        "FRP",
        "ART_CLEART",  # Clé de l'article
        "ART_NAART",  # Nature de l'article
        "ART_NAPMC",  # Nature mesure conservatoire
        "ART_DORI",  # Date d'origine
        "ART_DIDR",  # Date d'exibilité de l'impot
        "ART_DATEDCF",  # Date prise en compte de la notification de redressement (cf)
        "ART_DISC",  # Date d'inscription en RAR
        "ART_ICIRREL",  # Indicateur de circuit de relance" (souvent pas rempli)
        "MNT_CREANCE",  # Montant de la créance,
        "MVT_DJC",  # Date journée compatble
        "MNT_PAIEMENT_CUM",  # Montant des paiements à la date de la journée compatable
        "MNT_PAIEMENT_CUM_AUTRE",  # Montant des paiements dit "autres" à la djc.
    ]
)

x_creances = x_creances.withColumn(
    "IND_CF", F.when(F.col("ART_DATEDCF").isNotNull(), 1).otherwise(0)
)
x_creances = x_creances.withColumn(
    "IND_HCF", F.when(F.col("ART_DATEDCF").isNotNull(), 0).otherwise(1)
)

# TODO review how these operations handle null values.
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

# TVA can be declared either on a:
# - monthly,
# - quarterly,
# - yearly,
# basis
#
# We add the "DUREE_PERIODE_TVA" variable to describe this parameter.
#
ca3 = datasets["liasse_tva_ca3"].withColumn(
    "DUREE_PERIODE_TVA",
    F.round(
        F.months_between(
            datasets["liasse_tva_ca3"]["DTE_FIN_PERIODE"],
            datasets["liasse_tva_ca3"]["DTE_DEBUT_PERIODE"],
        )
    ).cast("integer"),
)

x_tva = ca3.withColumn("D_TCA_TOTAL", F.col("D3310_29") + F.col("D3517S_55_I"))
x_tva = x_tva.withColumn(
    "D_TVA_NI_B0032_EXPORT", F.col("D3517S_02_B") + F.col("D3310_04")
)
x_tva = x_tva.withColumn("D_TVA_NI_B0034_LIC", F.col("D3517S_04_B") + F.col("D3310_06"))
x_tva = x_tva.withColumn(
    "D_TVA_NI_B0037_ACH_FRCH", F.col("D3517S_01_B") + F.col("D3310_07")
)
x_tva = x_tva.withColumn(
    "D_TVA_NI_B0029_LIV_EL_GAZ", F.col("D3517S_4D_B") + F.col("D3310_6A")
)
x_tva = x_tva.withColumn(
    "D_TVA_NI_B0043_ASSJT_HS_FR", F.col("D3517S_4B_B") + F.col("D3310_7A")
)
x_tva = x_tva.withColumn(
    "M_TVA_NI_B0033_AUTR_OP_NI",
    F.col("D3310_7B") + F.col("D3517S_03_B") + F.col("D3310_05"),
)
cols_SUM_TVA_NI_bTOTAL = [
    "D_TVA_NI_B0032_EXPORT",
    "D_TVA_NI_B0034_LIC",
    "D_TVA_NI_B0037_ACH_FRCH",
    "D_TVA_NI_B0029_LIV_EL_GAZ",
    "D_TVA_NI_B0043_ASSJT_HS_FR",
    "M_TVA_NI_B0033_AUTR_OP_NI",
]
x_tva = x_tva.withColumn(
    "SUM_TVA_NI_BTOTAL", sum(x_tva[col] for col in cols_SUM_TVA_NI_bTOTAL)
)
cols_M_TVA_BI_b0979_CA = [
    "D3310_01",  # "D3517S_05_B0206",
    "D3517S_5A_B",
    "D3517S_06_B",  # "D3517S_6B_B0150",
    "D3517S_6C_B",
    "D3517S_07_B",
    "D3517S_08_B",
    "D3517S_09_B",
    "D3517S_10_B",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0979_CA", sum(x_tva[col] for col in cols_M_TVA_BI_b0979_CA)
)
cols_M_TVA_BI_b0981_AUTR_OP_IMP = [
    "D3310_02",  # "D3310_2B",
    "D3310_3C",
    "D3517S_13_B",
    "D3517S_11_B",
    "D3517S_12_B",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0981_AUTR_OP_IMP",
    sum(x_tva[col] for col in cols_M_TVA_BI_b0981_AUTR_OP_IMP),
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_B0044_ACH_PS_IC", F.col("D3517S_AC_B") + F.col("D3310_2A")
)
x_tva = x_tva.withColumn("D_TVA_BI_B0031_AIC", F.col("D3517S_14_B") + F.col("D3310_03"))
x_tva = x_tva.withColumn(
    "D_TVA_BI_B0030_LIV_EL_GAZ", F.col("D3517S_AA_B") + F.col("D3310_3A")
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_B0040_ASSJT_HS_FR", F.col("D3517S_AB_B") + F.col("D3310_3B")
)
cols_SUM_TVA_BI_bTOTAL = [
    "M_TVA_BI_B0979_CA",
    "M_TVA_BI_B0981_AUTR_OP_IMP",
    "D_TVA_BI_B0044_ACH_PS_IC",
    "D_TVA_BI_B0031_AIC",
    "D_TVA_BI_B0030_LIV_EL_GAZ",
    "D_TVA_BI_B0040_ASSJT_HS_FR",
]
x_tva = x_tva.withColumn(
    "SUM_TVA_BI_BTOTAL", sum(x_tva[col] for col in cols_SUM_TVA_BI_bTOTAL)
)
x_tva = x_tva.withColumn(
    "SUM_TVA_NI_BI_BTOTAL", F.col("SUM_TVA_BI_BTOTAL") + F.col("SUM_TVA_NI_BTOTAL")
)
cols_M_TVA_BI_b0207_NORMAL = [  # "D3517S_05_B0206",
    "D3310_08_BTX196",
    "D3517S_5A_B",
    "D3310_08_B",
    "D3517S_11_B",
    "D3517S_12_B",
    "D3517S_13_B",
    "D3517S_14_B",
    "D3517S_AB_B",
    "D3517S_AC_B",
    "D3517S_AA_B",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0207_NORMAL", sum(x_tva[col] for col in cols_M_TVA_BI_b0207_NORMAL)
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0105_REDUIT_5_5", F.col("D3517S_06_B") + F.col("D3310_09_B")
)
cols_M_TVA_BI_b0151_REDUIT_10 = [  # D3517S_6B_b0150,
    "D3310_9B_BTX7",
    "D3517S_6C_B",
    "D3310_9B_B",
]
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0151_REDUIT_10", sum(x_tva[col] for col in cols_M_TVA_BI_b0151_REDUIT_10)
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0100_DOM_2_1", F.col("D3517S_08_B") + F.col("D3310_11_B")
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0201_DOM_8_5", F.col("D3517S_07_B") + F.col("D3310_10_B")
)
x_tva = x_tva.withColumn(
    "D_TVA_BI_B0950_TX_PART", F.col("D3517S_09_B") + F.col("D3310_14_B")
)
x_tva = x_tva.withColumn(
    "M_TVA_BI_B0900_ANC_TX", F.col("D3517S_10_B") + F.col("D3310_13_B")
)
x_tva = x_tva.withColumn(
    "D_TVA_COL_I0600_ANT_DED", F.col("D3310_15") + F.col("D3517S_18_I")
)
x_tva = x_tva.withColumn(
    "SUM_TVA_COL_TOTAL",
    F.col("D3310_16")
    - F.col("D3310_15")
    + F.col("D3517S_16_I")
    - F.col("D3310_7C")
    - F.col("D3310_17")
    - F.col("D3310_5B")
    - F.col("D3517S_AA_I")
    - F.col("D3517S_AB_I")
    - F.col("D3517S_AC_I")
    - F.col("D3517S_13_I")
    - F.col("D3517S_14_I"),
)
x_tva = x_tva.withColumn(
    "D_TVA_COL_I0031_AIC", F.col("D3517S_14_I") + F.col("D3310_17")
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_I0703_IMM", F.col("D3310_19") + F.col("D3517S_23_I")
)
x_tva = x_tva.withColumn(
    "M_TVA_DED_I0702_ABS",
    F.col("D3310_20") + F.col("D3517S_20_I") + F.col("D3517S_21_I"),
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_I0059_AUTR", F.col("D3310_21") + F.col("D3517S_25_I")
)  # D3310_2C
# TODO if (D3310_22A=. AND D3517S_25A_TX_DED=.) then D_TVA_DED_TX_COEF_DED = 100; else
# D_TVA_DED_TX_COEF_DED = SUM(D3310_22A,D3517S_25A_TX_DED) ;
x_tva = x_tva.withColumn(
    "D_TVA_DED_I0705_TOTAL", F.col("D3310_23") + F.col("D3517S_26_I")
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_TOTAL_HS_REPORT",
    F.col("D_TVA_DED_I0703_IMM")
    + F.col("M_TVA_DED_I0702_ABS")
    + F.col("D_TVA_DED_I0059_AUTR"),
)
x_tva = x_tva.withColumn(
    "D_TVA_DED_I0709_DT_ES_DOM", F.col("D3310_24") + F.col("D3517S_27_I")
)
x_tva = x_tva.withColumn(
    "M_TVA_NET_I8002_REMB_DEM", F.col("D3310_26") + F.col("D3517S_50_I")
)
x_tva = x_tva.withColumn("M_TVA_NET_DUE", F.col("D3310_28") + F.col("D3517S_28_I"))

# Write file
rar_mois_article.write.format("orc").save(path.join(args.output, "rar"))
x_tva.write.format("orc").save(path.join(args.output, "tva"))
