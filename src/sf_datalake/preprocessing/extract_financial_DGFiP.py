"""Extract yearly DGFiP financial data.

Source datasets are the following csv files:
- etl_decla-declarations_indmap
- etl_decla-declarations_af
- ratios_dirco

A yearly dataset with some pre-computed + feature engineered ratios will be stored as
orc files under the chosen output directory.

USAGE
    python yearly_dgfip_dataset.py --help

to get more info on expected args.

"""
import datetime as dt
import os
import sys
from os import path
from typing import List

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import Window

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint:disable=wrong-import-position
import dateutil.parser
import pandas as pd

import sf_datalake.configuration
import sf_datalake.io
import sf_datalake.utils

####################
# Loading datasets #
####################

spark = sf_datalake.utils.get_spark_session()
parser = sf_datalake.io.data_path_parser()
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
parser.add_argument(
    "-c",
    "--configuration",
    help="""
    Configuration file. This will be used to fetch required features.
    """,
    required=True,
)
parser.add_argument(
    "--min_date",
    help="""Requested start date for the output dataset.""",
    default="2014-01-01",
)
parser.description = "Build a dataset of yearly DGFiP data."
args = parser.parse_args()

configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)
data_paths = {
    "indmap": path.join(args.input, "etl_decla", "declarations_indmap.csv"),
    "af": path.join(args.input, "etl_decla", "declarations_af.csv"),
    "dirco": path.join(args.input, "etl_rspro", "ratios_dirco.csv"),
    # "rar_tva": path.join(args.input, "cfvr", "rar_tva_exercice.csv"),
}
datasets = sf_datalake.io.load_data(
    data_paths, file_format="csv", sep="|", infer_schema=False
)

### Parse and clean file
# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

###################
# Merge datasets  #
###################

extract_dict = {
    "dirco": {"rto_6", "rto_56"},
    "af": {
        "mnt_af_bfonc_actif_circ_expl",
        "mnt_af_bfonc_actif_circ_h_expl",
        "mnt_af_bfonc_bfr",
        "mnt_af_bfonc_passif_circ_expl",
        "mnt_af_bfonc_passif_circ_h_expl",
        "mnt_af_bfonc_tresorerie",
        "mnt_af_ca",
        "mnt_af_endettement_net",
        "mnt_af_sig_ebe_ret",
        "mnt_af_sig_va_ret",
        "nbr_af_jours_creance_cli",
        "nbr_af_jours_reglt_fourn",
        "rto_af_endettement_a_terme",
        "rto_af_rent_eco",
    },
    "indmap": {
        "d_actf_stk_march_net",
        "d_actf_stk_mat1e_net",
        "d_cr_250_expl_salaire",
        "d_cr_252_expl_ch_soc",
        "d_cr_260_expl_dt_syndic",
        "d_dvs_376_nbr_pers",
        "d_passf_120_k",
        "d_passf_142_k_propres",
        "rto_invest_ca",
        "rto_af_solidite_financiere",
    },
}

# Join keys, as recommended by data providers, see SJCF-1D confluence.
decla_common_columns = set(datasets["af"].columns) & set(datasets["indmap"].columns)
decla_join_columns = {
    "siren",
    "date_deb_exercice",
    "date_fin_exercice",
    "no_ocfi",
    "annee_exercice",
}
decla_drop_columns = decla_common_columns - set(decla_join_columns)

# Combine tables
df = (
    datasets["indmap"]
    .join(
        datasets["af"].drop(*decla_drop_columns),
        on=list(decla_join_columns),
        how="inner",
    )
    .select(*(decla_join_columns | extract_dict["indmap"] | extract_dict["af"]))
    .join(
        datasets["dirco"].select(
            *(
                {"siren", "date_deb_exercice", "date_fin_exercice"}
                | extract_dict["dirco"]
            )
        ),
        on=["siren", "date_deb_exercice", "date_fin_exercice"],
        how="left",
    )
)
# # Join TVA annual debt data
# df = declarations.join(
#     datasets["rar_tva"], on=list(join_columns - {"no_ocfi"}), how="left"
# )

### Rename and transform date columns
df = df.withColumnRenamed("annee_exercice", "année_exercice").withColumn(
    "année_exercice", F.col("année_exercice").cast("int")
)
df = df.withColumnRenamed("date_deb_exercice", "date_début_exercice").withColumn(
    "date_début_exercice", F.to_date("date_début_exercice")
)
df = df.withColumn("date_fin_exercice", F.to_date("date_fin_exercice"))

# Point out which variables are used as source for feature computation
feature_cols: List[str] = configuration.explanation.topic_groups.get("santé_financière")
source_variables: List[str] = [
    "mnt_af_endettement_net",
    "rto_6",
    "rto_af_endettement_a_terme",
    "mnt_af_sig_ebe_ret",
    "mnt_af_ca",
    "mnt_af_sig_va_ret",
    "d_dvs_376_nbr_pers",
    "d_cr_250_expl_salaire",
    "d_cr_252_expl_ch_soc",
    "d_cr_260_expl_dt_syndic",
    "d_actf_stk_march_net",
    "mnt_af_bfonc_actif_circ_expl",
    "mnt_af_bfonc_actif_circ_h_expl",
    "mnt_af_bfonc_passif_circ_expl",
    "mnt_af_bfonc_passif_circ_h_expl",
    "mnt_af_bfonc_tresorerie",
    "nbr_af_jours_reglt_fourn",
    "nbr_af_jours_creance_cli",
    "d_passf_120_k",
    "mnt_af_bfonc_bfr",
    "d_passf_142_k_propres",
]

# Filter by date
df = df.filter(F.col("date_fin_exercice") > dateutil.parser.parse(args.min_date))

# We remove data where multiple declarations exist for a given (SIREN, date) couple and
# only keep the line with the lowest null values count.

# Create a monthly date range that will become a time index
date_range = spark.createDataFrame(
    pd.DataFrame(
        pd.date_range(args.min_date, dt.date.today().isoformat(), freq="MS")
        .to_series()
        .dt.date,
        columns=["période"],
    )
)

df = df.join(
    date_range,
    on=date_range["période"].between(
        df["date_début_exercice"], F.date_sub(df["date_fin_exercice"], 1)
    ),
    how="inner",
)
df = df.withColumn(
    "null_count",
    sum([F.when(F.col(c).isNull(), 1).otherwise(0) for c in df.columns]),
)
w = Window().partitionBy(["siren", "période"]).orderBy(F.col("null_count").asc())
df = (
    df.withColumn("n_row", F.row_number().over(w))
    .filter(F.col("n_row") == 1)
    .drop("n_row")
)

########################
# Feature engineering  #
########################

# Handle missing values for variables that are used in the following computation
df = sf_datalake.transform.MissingValuesHandler(
    inputCols=source_variables,
    # TODO: Make this mapping better if needed
    value={var: 0.0 for var in source_variables},
).transform(df)


df = df.withColumn(
    "dette_nette_sur_caf",
    df["mnt_af_endettement_net"] / df["rto_6"],
)
df = df.withColumn("dette_à_terme_sur_k_propres", 1 / df["rto_af_endettement_a_terme"])
df = df.withColumn("ebe_ca", df["mnt_af_sig_ebe_ret"] / df["mnt_af_ca"])
df = df.withColumn(
    "va_sur_effectif",
    df["mnt_af_sig_va_ret"] / df["d_dvs_376_nbr_pers"],
)
df = df.withColumn(
    "charges_personnel_surva",
    (
        df["d_cr_250_expl_salaire"]
        + df["d_cr_252_expl_ch_soc"]
        + df["d_cr_260_expl_dt_syndic"]
    )
    / df["mnt_af_sig_va_ret"],
)
df = df.withColumn("stocks_sur_ca", df["d_actf_stk_march_net"] / df["mnt_af_ca"])
df = df.withColumn(
    "liquidité_absolue",
    (df["mnt_af_bfonc_actif_circ_expl"] + df["mnt_af_bfonc_actif_circ_h_expl"])
    / (df["mnt_af_bfonc_passif_circ_expl"] + df["mnt_af_bfonc_passif_circ_h_expl"]),
)
df = df.withColumn(
    "liquidité_générale",
    df["mnt_af_bfonc_tresorerie"]
    / (df["mnt_af_bfonc_actif_circ_expl"] + df["mnt_af_bfonc_actif_circ_h_expl"]),
)
df = df.withColumn(
    "délai_paiement_sur_délai_encaissement",
    (df["nbr_af_jours_reglt_fourn"] / df["nbr_af_jours_creance_cli"]),
)
df = df.withColumn(
    "k_propres_sur_k_social",
    (df["d_passf_142_k_propres"] / df["d_passf_120_k"]),
)
df = df.withColumn(
    "bfr_sur_k_propres",
    (df["mnt_af_bfonc_bfr"] / df["d_passf_142_k_propres"]),
)

############
# Cleaning #
############

# Rename to readable french
df = df.withColumnRenamed("rto_invest_ca", "taux_investissement")
df = df.withColumnRenamed("rto_af_solidite_financiere", "solidité_financière")
df = df.withColumnRenamed("rto_56", "liquidité_réduite")
df = df.withColumnRenamed("rto_af_rent_eco", "rentabilité_économique")

# Drop features that were only used for feature engineering.
df = df.drop(*source_variables)

################################
# Preprocess computed features #
################################

time_normalizer = sf_datalake.transform.TimeNormalizer(
    inputCols=feature_cols,
    start="date_début_exercice",
    end="date_fin_exercice",
)
# sf_datalake.transform.TimeNormalizer(
#     inputCols=[""], start="date_deb_tva", end="date_fin_tva"
# )

mvh_fe = sf_datalake.transform.MissingValuesHandler(
    inputCols=feature_cols,
    value=configuration.preprocessing.fill_default_values,
)

df = PipelineModel([time_normalizer, mvh_fe]).transform(df)

##########
# Export #
##########


sf_datalake.io.write_data(
    df.select(
        feature_cols
        + ["siren", "date_début_exercice", "date_fin_exercice", "no_ocfi", "période"]
    ),
    args.output,
    args.output_format,
)
