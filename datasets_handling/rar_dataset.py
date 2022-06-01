"""Build a dataset of rar ("reste à recouvrer") data.

This roughly follows MRV's process, originally written in SAS
`21_indicateurs.sas`. Source data should be stored beforehand inside an input directory
which, in turn, contains the following directories containing the data as (possibly
multiple) orc file(s):
- t_art (pub_risq_oracle)
- t_mvt (pub_risq_oracle)
- etl_refent-T_ETABLISSEMENT_ANNEE

USAGE
    python rar_dataset.py <pub_risq_tables_directory> <output_directory>

"""

import os
import sys
from os import path

import pyspark
import pyspark.sql
import pyspark.sql.functions as F

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable = C0413
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils


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
    output_cols = {"mnt": "mnt_paiement_cum", "nb": "nb_paiement"}
    if suffix is not None:
        output_cols = {key: f"{col}_{suffix}" for key, col in output_cols.items()}
    window = (
        pyspark.sql.Window.partitionBy("art_cleart")
        .orderBy(["frp", "min(mvt_djc_int)"])
        .rangeBetween(pyspark.sql.Window.unboundedPreceding, 0)
    )

    df = df.withColumn("mvt_djc_int", F.unix_timestamp(F.col("mvt_djc")))
    gb = df.orderBy("frp", "art_cleart", "mvt_djc").groupBy(
        ["frp", "art_cleart", "mvt_deff"]
    )
    df_agg = (
        gb.agg(F.min("mvt_djc_int"), F.sum("mvt_mcrd"))
        .select(["frp", "art_cleart", "min(mvt_djc_int)", "sum(mvt_mcrd)"])
        .dropDuplicates()
    )
    return (
        df_agg.filter(F.col("sum(mvt_mcrd)") != 0)
        .withColumn(
            output_cols["mnt"],
            F.sum("sum(mvt_mcrd)").over(window),
        )
        .withColumn(output_cols["nb"], F.count("sum(mvt_mcrd)").over(window))
        .dropDuplicates()
        .drop(*["sum(mvt_mcrd)", "min(mvt_djc_int)"])
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
    "t_etablissement_annee": path.join(args.input, "etl_refent-T_ETABLISSEMENT_ANNEE"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="orc")

# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

    #######
# RAR #
#######

# Parse dates, create "frp" join key

t_art = sf_datalake.transform.parse_date(
    datasets["t_art"], ["art_disc", "art_didr", "art_datedcf", "art_dori"]
)
t_mvt = sf_datalake.transform.parse_date(datasets["t_mvt"], ["mvt_djc", "mvt_deff"])

corresp_siren_frp2 = (
    datasets["t_etablissement_annee"]
    .withColumn(
        "frp",
        F.concat(
            datasets["t_etablissement_annee"]["frp_service"],
            datasets["t_etablissement_annee"]["frp_dossier"],
        ),
    )
    .drop(*["frp_service", "frp_dossier"])
)

mvt_montant_creance = t_mvt.join(
    t_mvt.groupBy(["frp", "art_cleart"])
    .sum("mvt_mdb")
    .withColumnRenamed("sum(mvt_mdb)", "mnt_creance"),
    on=["frp", "art_cleart"],
    how="left",
).drop(*["id", "date_chargement", "frp_service", "frp_dossier"])

mvt_paiement_nacrd01 = process_payment(t_mvt.filter("mvt_nacrd == 0 OR mvt_nacrd == 1"))
mvt_paiement_nacrd_autre = process_payment(
    t_mvt.filter("mvt_nacrd != 0 AND mvt_nacrd != 1"), suffix="autre"
)

# Join all tables
creances = (
    t_art.join(mvt_montant_creance, on=["frp", "art_cleart"], how="left")
    .join(
        mvt_paiement_nacrd01.drop(*["frp_service", "frp_dossier"]),
        on=["frp", "art_cleart"],
        how="left",
    )
    .join(
        mvt_paiement_nacrd_autre.drop(*["frp_service", "frp_dossier"]),
        on=["frp", "art_cleart"],
        how="left",
    )
    .join(corresp_siren_frp2, on=["frp"], how="left")
)

x_creances = creances.select(
    [
        "siren",
        "frp",
        "art_cleart",  # Clé de l'article
        "art_naart",  # Nature de l'article
        "art_napmc",  # Nature mesure conservatoire
        "art_dori",  # Date d'origine
        "art_didr",  # Date d'exigibilité de l'impot
        "art_datedcf",  # Date prise en compte de la notification de redressement (cf)
        "art_disc",  # Date d'inscription en RAR
        "art_icirrel",  # Indicateur de circuit de relance (souvent pas rempli)
        "mnt_creance",  # Montant de la créance,
        "mvt_djc",  # Date journée compatble
        "mnt_paiement_cum",  # Montant des paiements à la date de la journée compatable
        "mnt_paiement_cum_autre",  # Montant des paiements dit "autres" à la djc.
    ]
).na.fill(value=0, subset=["mnt_creance", "mnt_paiement_cum", "mnt_paiement_cum_autre"])

x_creances = x_creances.withColumn(
    "ind_cf", F.when(F.col("art_datedcf").isNotNull(), 1).otherwise(0)
)
x_creances = x_creances.withColumn(
    "ind_hcf", F.when(F.col("art_datedcf").isNotNull(), 0).otherwise(1)
)

rar_mois_article = x_creances.withColumn(
    "mnt_paiement_cum_tot",
    sum(x_creances[col] for col in ["mnt_paiement_cum", "mnt_paiement_cum_autre"]),
)
rar_mois_article = rar_mois_article.withColumn(
    "mnt_paiement_cum_tot_hcf", F.col("mnt_paiement_cum_tot") * F.col("ind_hcf")
)
rar_mois_article = rar_mois_article.withColumn(
    "mnt_creance_hcf", F.col("mnt_creance") * F.col("ind_hcf")
)
rar_mois_article = rar_mois_article.withColumn(
    "mnt_rar", F.col("mnt_creance") - F.col("mnt_paiement_cum_tot")
)
rar_mois_article = rar_mois_article.withColumn(
    "mnt_rar_hcf", F.col("mnt_rar") * F.col("ind_hcf")
)
rar_mois_article.write.format("orc").save(path.join(args.output, "rar"))
