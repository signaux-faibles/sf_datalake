"""Build a dataset of yearly DGFiP data.

This follows MRV's process, originally written in SAS. Source data should be stored
beforehand inside an input directory which, in turn, contains 3 directories containing
the data as (possibly multiple) orc file(s):
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

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.io

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset of yearly DGFiP data."
args = parser.parse_args()

data_paths = {
    "indmap": path.join(args.input, "etl_decla-declarations_indmap"),
    "af": path.join(args.input, "etl_decla-declarations_af"),
    "rar_tva": path.join(args.input, "rar_tva_exercice"),
}
datasets = sf_datalake.io.load_data(data_paths, file_format="orc")

# Set every column name to lower case (if not already).
for name, ds in datasets.items():
    datasets[name] = ds.toDF(*(col.lower() for col in ds.columns))

###################
# Merge datasets  #
###################

join_columns = {"siren", "date_deb_exercice", "date_fin_exercice", "no_ocfi"}
common_columns = set(datasets["af"].columns) & set(datasets["indmap"].columns)
drop_columns = common_columns - join_columns
# TODO: maybe we can drop less columns and still get a complete inner join,

# Combine 'declarations' tables
declarations = datasets["indmap"].join(
    datasets["af"].drop(*drop_columns),
    on=list(join_columns),
    how="inner",
)

# Join TVA annual debt data
df = declarations.join(
    datasets["rar_tva"], on=list(join_columns - {"no_ocfi"}), how="left"
)

df.write.format("orc").save(args.output)
