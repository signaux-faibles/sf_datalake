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
import os
import sys
from os import path
from typing import List

from pyspark.ml import PipelineModel

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint:disable=wrong-import-position
import sf_datalake.configuration
import sf_datalake.io

####################
# Loading datasets #
####################

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

# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

### TODO cast columns to the right type
# str: siren, no_ocfi
# dates: dates début, date fin
# int : année exercice,
#

### TODO : filter by date


###################
# Merge datasets  #
###################

# Join keys, as recommended by data providers, see SJCF-1D confluence.
common_columns = set(datasets["af"].columns) & set(datasets["indmap"].columns)
drop_columns = common_columns - {
    "siren",
    "date_deb_exercice",
    "date_fin_exercice",
    "no_ocfi",
}

# Combine tables
df = (
    datasets["indmap"]
    .join(
        datasets["af"].drop(*drop_columns),
        on={"siren", "date_deb_exercice", "date_fin_exercice", "no_ocfi"},
        how="inner",
    )
    .join(
        datasets["dirco"],
        on={"siren", "date_deb_exercice", "date_fin_exercice"},
        how="left",
    )
)
# # Join TVA annual debt data
# df = declarations.join(
#     datasets["rar_tva"], on=list(join_columns - {"no_ocfi"}), how="left"
# )

df = df.withColumnRenamed("mnt_af_bfonc_bfr", "bfr")
df = df.withColumnRenamed("rto_invest_ca", "taux_investissement")
df = df.withColumnRenamed("rto_af_rent_eco", "rentabilité_économique")
df = df.withColumnRenamed("rto_af_solidite_financiere", "solidité_financière")
df = df.withColumnRenamed("rto_56", "liquidité_réduite")
df = df.withColumnRenamed("rto_invest_ca", "taux_investissement")
df = df.withColumnRenamed("rto_af_rent_eco", "rentabilité_économique")
df = df.withColumnRenamed("rto_af_solidite_financiere", "solidité_financière")


df = df.withColumnRenamed("date_deb_exercice", "date_début_exercice")

feature_cols: List[str] = configuration.explanation.topic_groups.get("santé_financière")
engineered_feature_cols: List[str] = [
    "bfr/k_propres",
    "k_propres/k_social",
    "délai_paiement/délai_encaissement",
    "liquidité_générale",
    "liquidité_absolue",
    "stocks/ca",
    "charges_personnel/va",
    "va/effectif",
    "ebe/ca",
    "dette_à_terme/k_propres",
    "dette_nette/caf",
]
base_feature_cols: List[str] = list(set(feature_cols)) - list(
    set(engineered_feature_cols)
)

#################
# Preprocess raw features #
#################

df = sf_datalake.transform.MissingValuesHandler(
    inputCols=base_feature_cols,
    value=configuration.preprocessing.fill_default_values,
).transform(df)

########################
# Feature engineering  #
########################

df = df.withColumn(
    "dette_nette/caf",
    df["mnt_af_endettement_net"] / df["rto_6"],
)
df = df.withColumn("dette_à_terme/k_propres", 1 / df["rto_af_endettement_a_terme"])
df = df.withColumn("ebe/ca", df["mnt_af_sig_ebe_ret"] / df["mnt_af_ca"])
df = df.withColumn(
    "va/effectif",
    df["mnt_af_sig_va_ret"] / df["d_dvs_376_nbr_pers"],
)
df = df.withColumn(
    "charges_personnel/va",
    (
        df["d_cr_250_expl_salaire"]
        + df["d_cr_252_expl_ch_soc"]
        + df["d_cr_260_expl_dt_syndic"]
    )
    / df["mnt_af_sig_va_ret"],
)
df = df.withColumn("stocks/ca", df["d_actf_stk_march_net"] / df["mnt_af_ca"])
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
    "délai_paiement/délai_encaissement",
    (df["nbr_af_jours_reglt_fourn"] / df["nbr_af_jours_creance_cli"]),
)

df = df.withColumn(
    "k_propres/k_social",
    (df["d_passf_142_k_propres"] / df["d_passf_120_k"]),
)

df = df.withColumn(
    "bfr/k_propres",
    (df["mnt_af_bfonc_bfr"] / df["d_passf_142_k_propres"]),
)

#################
# Preprocess computed features #
#################

time_normalizer = [
    sf_datalake.transform.TimeNormalizer(
        inputCols=feature_cols,
        start="date_début_exercice",
        end="date_fin_exercice",
    ),
    # sf_datalake.transform.TimeNormalizer(
    #     inputCols=[""], start="date_deb_tva", end="date_fin_tva"
    # ),
]

mvh_fe = sf_datalake.transform.MissingValuesHandler(
    inputCols=engineered_feature_cols,
    value=configuration.preprocessing.fill_default_values,
)

df = PipelineModel([time_normalizer, mvh_fe]).transform(df)

##########
# Export #
##########


sf_datalake.io.write_data(
    df.select(
        feature_cols
        + list({"siren", "date_début_exercice", "date_fin_exercice", "no_ocfi"})
    ),
    args.output,
    args.output_format,
)
