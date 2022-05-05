"""Build a dataset of various data aggregated at the SIREN level.

An output dataset will be stored as split orc files under the chosen output directory.

USAGE
    python aggregate_to_siren.py <input_dataset_directory> <output_directory>

"""
import os
import sys
from os import path

from pyspark.ml import PipelineModel

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.io
import sf_datalake.transform

####################
# Loading datasets #
####################

parser = sf_datalake.io.data_path_parser()
parser.description = (
    "Build a dataset of SIREN-aggregated data from SIRET-level variables."
)

args = parser.parse_args()

data_paths = {"input_ds": args.input}
siret_level_ds = sf_datalake.io.load_data(data_paths, file_format="orc")["input_ds"]

# Set every column name to lower case (if not already).
siret_level_ds = siret_level_ds.toDF(*(col.lower() for col in siret_level_ds.columns))

# Generate a clean SIREN column
siret_level_ds = sf_datalake.transform.extract_siren_from_siret(siret_level_ds)

#####################
# Make aggregation  #
#####################

# Filter out public institutions and companies and aggregate at SIREN level
naf_filter = sf_datalake.transform.PrivateCompanyFilter()
aggregator = sf_datalake.transform.SirenAggregator(
    sf_datalake.io.load_variables("aggregation.json")
)
siren_level_ds = PipelineModel(stages=[naf_filter, aggregator]).transform(
    siret_level_ds
)
siren_level_ds.write.format("orc").save(args.output)
