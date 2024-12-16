"""Build a dataset by joining data from various sources along time and SIREN.

Expected inputs :
- URSSAF debit data
- URSSAF cotisation data
- DGEFP data
- INSEE 'sirene' administrative data
- INSEE 'sirene' activity dates data
- altares 'paydex' + 'FPI' data
- DGFiP financial ratios dataset
- DGFiP judgment data

Inputs are expected to be folders containing ORC files, except for the following
sources, where a single CSV is expected:
- "sirene_categories"
- "sirene_dates"
- "effectif"

The time index column should be named 'période' and formatted as follows : "yyyy-MM-dd"

Type
  python join_datasets.py --help
for detailed usage.

"""
import argparse
import os
import sys
from os import path

import pyspark.sql.functions as F
import pyspark.sql.types as T

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.transform
import sf_datalake.utils
from sf_datalake.io import load_data, write_data

spark = sf_datalake.utils.get_spark_session()

parser = argparse.ArgumentParser(
    description="Merge DGFiP and Signaux Faibles datasets into a single one."
)
parser.add_argument(
    "--effectif",
    dest="effectif",
    help="Path to the preprocessed 'URSSAF effectif_ent' dataset.",
)
parser.add_argument(
    "--urssaf_debit",
    dest="urssaf_debit",
    help="Path to the preprocessed 'URSSAF debit' dataset.",
)
parser.add_argument(
    "--urssaf_cotisation",
    dest="urssaf_cotisation",
    help="Path to the preprocessed 'URSSAF cotisation' dataset.",
)
parser.add_argument(
    "--ap",
    dest="ap",
    help="Path to the preprocessed DARES dataset.",
)
parser.add_argument(
    "--sirene_categories",
    help="Path to the preprocessed sirene categorical dataset.",
)
parser.add_argument(
    "--sirene_dates",
    help="Path to the sirene companies activity dates.",
)
parser.add_argument(
    "--judgments",
    help="Path to the preprocessed judgments dataset.",
)
parser.add_argument(
    "--altares",
    help="Path to the preprocessed altares dataset.",
)
parser.add_argument(
    "--dgfip_yearly",
    help="Path to the DGFiP yearly dataset.",
)
parser.add_argument(
    "--perimeter",
    help="Path to the SIREN perimeter data.",
    required=False,
)
parser.add_argument(
    "--output_path",
    help="Output dataset directory path.",
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)

args = parser.parse_args()

# Load datasets
datasets = load_data(
    {
        "urssaf_debit": args.urssaf_debit,
        "urssaf_cotisation": args.urssaf_cotisation,
        "ap": args.ap,
        "dgfip_yearly": args.dgfip_yearly,
        "judgments": args.judgments,
        "altares": args.altares,
    },
    file_format="orc",
)

sirene_dates_schema = T.StructType(
    [
        T.StructField("siren", T.StringType(), False),
        T.StructField("date_fin", T.DateType(), True),
        T.StructField("date_début", T.DateType(), True),
    ]
)

sirene_categories_schema = T.StructType(
    [
        T.StructField("siren", T.StringType(), False),
        T.StructField("siret", T.StringType(), True),
        T.StructField("code_commune", T.StringType(), True),
        T.StructField("code_naf", T.StringType(), True),
        T.StructField("région", T.StringType(), True),
        T.StructField("catégorie_juridique", T.StringType(), True),
    ]
)
effectif_schema = T.StructType(
    [
        T.StructField("siren", T.StringType(), False),
        T.StructField("période", T.DateType(), False),
        T.StructField("effectif", T.IntegerType(), True),
    ]
)
perimeter_schema = T.StructType([T.StructField("siren", T.StringType(), False)])

datasets["sirene_categories"] = spark.read.csv(
    args.sirene_categories, header=True, schema=sirene_categories_schema
)
datasets["sirene_dates"] = spark.read.csv(
    args.sirene_dates, header=True, schema=sirene_dates_schema
)
datasets["effectif"] = spark.read.csv(
    args.effectif, header=True, schema=effectif_schema
)

# Prepare datasets
siren_normalizer = sf_datalake.transform.IdentifierNormalizer(inputCol="siren")
df_dgfip_yearly = siren_normalizer.transform(datasets["dgfip_yearly"])
df_judgments = siren_normalizer.transform(datasets["judgments"])
df_altares = siren_normalizer.transform(datasets["altares"])
df_urssaf_debit = siren_normalizer.transform(datasets["urssaf_debit"])
df_urssaf_cotisation = siren_normalizer.transform(datasets["urssaf_cotisation"])
df_sirene_categories = siren_normalizer.transform(datasets["sirene_categories"])
df_sirene_dates = siren_normalizer.transform(datasets["sirene_dates"]).fillna(
    {"date_fin": "2100-01-01"}
)
df_ap = siren_normalizer.transform(datasets["ap"])
df_effectif = siren_normalizer.transform(datasets["effectif"])

# Only keep companies that have had a workforce of at least 10 for some time.
max_wf_by_siren = df_effectif.groupBy("siren").max("effectif")
wf_mask = max_wf_by_siren.filter(F.col("max(effectif)") >= 10)
df_effectif = df_effectif.join(wf_mask, on="siren", how="left_semi")

# Join datasets
monthly_df = (
    df_urssaf_cotisation.join(df_urssaf_debit, on=["siren", "période"], how="left")
    .join(df_effectif, on=["siren", "période"], how="inner")
    .join(df_ap, on=["siren", "période"], how="left")
    .join(df_judgments, on="siren", how="left")
    .join(df_altares, on=["siren", "période"], how="left")
    .join(df_sirene_categories, on="siren", how="inner")
)

# Join monthly dataset with yearly dataset
joined_df = sf_datalake.utils.merge_asof(
    monthly_df,
    df_dgfip_yearly,
    on="période",
    by="siren",
    tolerance=365,
    direction="backward",
)

# Filter dataset to perimeter (if provided), and known activity dates
if args.perimeter is not None:
    siren_perimeter = spark.read.csv(
        args.perimeter,
        header=True,
        schema=perimeter_schema,
    )
    joined_df = joined_df.join(siren_perimeter, on="siren", how="left_semi")

output_df = joined_df.join(
    df_sirene_dates,
    on=(
        (joined_df["siren"] == df_sirene_dates["siren"])
        & (joined_df["période"] >= df_sirene_dates["date_début"])
        & (joined_df["période"] < df_sirene_dates["date_fin"])
    ),
    how="left_semi",
)

write_data(output_df, args.output_path, args.output_format)
