"""Write some data to files.
"""

import logging
import os
import sys
from os import path

from pyspark.sql import SparkSession  # pylint: disable=E0401

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
from sf_datalake.preprocessing import DATA_ROOT_DIR
from sf_datalake.utils import load_data

### Launch spark session, load data

spark = SparkSession.builder.getOrCreate()

logging.info(
    "Reading data in %s", path.join(DATA_ROOT_DIR, "base/indicateurs_annuels.orc")
)
indics_annuels = load_data(
    {"indics_annuels": path.join(DATA_ROOT_DIR, "base/indicateurs_annuels.orc")}
)["indics_annuels"].select("siren", "MNT_AF_CA", "year")

output_folder = "/projets/TSF/donnees/test/"

distributed = indics_annuels.filter(indics_annuels["year"] < 2019)
repartition1 = indics_annuels.filter(indics_annuels["year"] >= 2019)

distributed.write.csv(path.join(output_folder, "dist.csv"), header=True)
repartition1.repartition(1).write.csv(path.join(output_folder, "rep1.csv"), header=True)
