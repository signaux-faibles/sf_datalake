"""Build a dataset of yearly DGFiP data.

Source data should be stored beforehand inside an input directory which, in turn,
contains 3 directories containing the data as (possibly multiple) orc file(s):
- etl_decla-declarations_indmap
- etl_decla-declarations_af
- rar.rar_tva_exercice

A yearly dataset will be stored as split orc files under the chosen output directory.

USAGE
    python make_yearly_data.py <DGFiP_tables_directory> <output_directory>

"""
import os
import sys
from os import path
from typing import List

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
    # "rar_tva": path.join(args.input, "cfvr", "rar_tva_exercice.csv"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="csv", sep="|")

# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

###################
# Merge datasets  #
###################

# Join keys, as recommended by data providers, see SJCF-1D confluence.
join_columns = {"siren", "date_deb_exercice", "date_fin_exercice", "no_ocfi"}
common_columns = set(datasets["af"].columns) & set(datasets["indmap"].columns)
drop_columns = common_columns - join_columns

# Combine 'declarations' tables
declarations = datasets["indmap"].join(
    datasets["af"].drop(*drop_columns),
    on=list(join_columns),
    how="inner",
)

# # Join TVA annual debt data
# df = declarations.join(
#     datasets["rar_tva"], on=list(join_columns - {"no_ocfi"}), how="left"
# )

feature_cols: List[str] = configuration.explanation.topic_groups.get("sante_financiere")
time_normalizer = [
    sf_datalake.transform.TimeNormalizer(
        inputCols=feature_cols,
        start="date_deb_exercice",
        end="date_fin_exercice",
    ),
    # sf_datalake.transform.TimeNormalizer(
    #     inputCols=[""], start="date_deb_tva", end="date_fin_tva"
    # ),
]


sf_datalake.io.write_data(
    declarations.select(feature_cols + list(join_columns)),
    args.output,
    args.output_format,
)
