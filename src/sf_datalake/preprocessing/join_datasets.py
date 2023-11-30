"""Build a dataset by joining data from various sources.

The join is made along temporal and SIREN variables. Source files are
expected to be ORC.

Expected inputs :
- URSSAF debit data
- URSSAF cotisation data
- DGEFP data
- 'sirene' database data
- altares 'paydex' + 'FPI' data
- DGFiP financial ratios dataset
- DGFiP judgment data

TimeIndex names need to be "période"
and format as follow : "yyyy-MM-dd"

Type python join_datasets.py --help for detailed usage.

"""
import argparse
import os
import sys
from os import path

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.transform
import sf_datalake.utils
from sf_datalake.io import load_data, write_data

parser = argparse.ArgumentParser(
    description="Merge DGFiP and Signaux Faibles datasets into a single one."
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
    "--sirene",
    dest="sirene",
    help="Path to the preprocessed Sirene dataset.",
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
        "sirene": args.sirene,
        "dgfip_yearly": args.dgfip_yearly,
        "judgments": args.judgments,
        "altares": args.altares,
    },
    file_format="orc",
)


# Prepare datasets
siren_normalizer = sf_datalake.transform.IdentifierNormalizer(inputCol="siren")
df_dgfip_yearly = siren_normalizer.transform(datasets["dgfip_yearly"])
df_judgments = siren_normalizer.transform(datasets["judgments"])
df_altares = siren_normalizer.transform(datasets["altares"])
df_urssaf_debit = siren_normalizer.transform(datasets["urssaf_debit"])
df_urssaf_cotisation = siren_normalizer.transform(datasets["urssaf_cotisation"])
df_sirene = siren_normalizer.transform(datasets["sirene"])
df_ap = siren_normalizer.transform(datasets["ap"])

# Join "monthly" datasets
joined_df_monthly = (
    df_urssaf_debit.join(df_urssaf_cotisation, on=["siren", "periode"], how="inner")
    .drop(df_urssaf_cotisation.siren)
    .join(df_ap, on=["siren", "periode"], how="inner")
    .join(df_sirene, on="siren", how="inner")
    .join(df_judgments, on="siren", how="left")
    .join(df_altares, on=["siren", "periode"], how="left")
)
# Rename "target" time index for merge asof
df_dgfip_yearly = df_dgfip_yearly.withColumnRenamed("date_deb_exercice", "periode")

# Join monthly dataset with yearly dataset
joined_df = sf_datalake.utils.merge_asof(
    joined_df_monthly,
    df_dgfip_yearly,
    on="periode",
    by="siren",
    tolerance=365,
    direction="backward",
)

write_data(joined_df, args.output_path, args.output_format)
