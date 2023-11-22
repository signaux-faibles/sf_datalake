"""Carry out some pre-processing over URSSAF "cotisation" data.

Run `python extract_cotisation_urssaf.py --help` to get usage insights.

The data is documented here:
https://github.com/signaux-faibles/documentation/blob/master/description-donnees.md\
#donn%C3%A9es-sur-les-cotisations-sociales-et-les-d%C3%A9bits

"""
# pylint: disable=duplicate-code
import datetime as dt
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
import pandas as pd

import sf_datalake.configuration
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

####################
# Loading datasets #
####################

spark = sf_datalake.utils.get_spark_session()
parser = sf_datalake.io.data_path_parser()
parser.description = "Build a dataset with aggregated SIREN-level data."
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
parser.add_argument("--min_date", default="2014-01-01")
parser.add_argument(
    "--configuration",
    help="""
    Configuration file name (including '.json' extension). If not provided,
    'standard.json' will be used.
    """,
    default="standard.json",
)
args = parser.parse_args()
configuration = sf_datalake.configuration.ConfigurationHelper(
    config_file=args.configuration
)

cotisation_schema = T.StructType(
    [
        T.StructField("siret", T.StringType(), False),
        T.StructField("numéro_compte", T.StringType(), True),
        T.StructField("fenêtre", T.StringType(), True),
        T.StructField("encaissé", T.DoubleType(), True),
        T.StructField("dû", T.DoubleType(), True),
    ]
)
siret_to_siren = sf_datalake.transform.SiretToSiren()

# Create a monthly date range that will become the time index
dr = pd.date_range(args.min_date, dt.date.today().isoformat(), freq="MS")
date_range = spark.createDataFrame(
    pd.DataFrame(dr.to_series().dt.date, columns=["période"])
)

## "Cotisation" data
cotisation = spark.read.csv(args.input, header=True, schema=cotisation_schema)

# Preprocess "fenêtre", which comes as two adjacent dates in the following format:
# "YYYY-MM-DDThh:mm:ss-YYYY-MM-DDThh:mm:ss"
cotisation = cotisation.dropna(subset="fenêtre")
cotisation = cotisation.withColumn(
    "date_début", F.to_date(F.substring(F.col("fenêtre"), 1, 10))
)
cotisation = cotisation.withColumn(
    "date_fin", F.to_date(F.substring(F.col("fenêtre"), 21, 10))
)
cotisation = cotisation.filter(F.col("date_fin") > args.min_date)

# Spread over the time periods
cotisation = siret_to_siren.transform(cotisation)
cotisation = cotisation.withColumn(
    "cotisation_appelée_par_mois",
    F.col("dû") / F.months_between("date_fin", "date_début"),
)

cotisation = cotisation.join(
    date_range,
    on=date_range["période"].between(
        cotisation["date_début"], F.date_sub(cotisation["date_fin"], 1)
    ),
    how="inner",
)

# Handle missing values and export
mvh = sf_datalake.transform.MissingValuesHandler(
    inputCols=["cotisation"],
    value=configuration.preprocessing.fill_default_values,
)

output_ds = mvh.transform(
    cotisation.groupBy(["siren", "période"]).agg(
        F.sum("cotisation_appelée_par_mois").alias("cotisation")
    )
)

sf_datalake.io.write_data(output_ds, args.output, args.output_format)
