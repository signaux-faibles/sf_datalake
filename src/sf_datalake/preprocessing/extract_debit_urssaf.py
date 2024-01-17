"""Carry out some pre-processing over URSSAF "débit" data.

Run `python extract_debit_urssaf.py --help` to get usage insights.

The data is documented here (variable names may be slightly different after some
upstream normalization):
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
from pyspark.sql import Window

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

debit_schema = T.StructType(
    [
        T.StructField("siret", T.StringType(), False),
        T.StructField("numéro_compte", T.StringType(), True),
        T.StructField("numéro_écart_négatif", T.IntegerType(), True),
        T.StructField("date_traitement", T.StringType(), False),
        T.StructField("dette_sociale_ouvrière", T.DoubleType(), True),
        T.StructField("dette_sociale_patronale", T.DoubleType(), True),
        T.StructField("numéro_historique_écart_négatif", T.ShortType(), True),
        T.StructField("état_compte", T.IntegerType(), True),
        T.StructField("code_procedure_collective", T.ByteType(), True),
        T.StructField("période_cotisation", T.StringType(), True),
        T.StructField("code_operation_ecart_negatif", T.ByteType(), True),
        T.StructField("code_motif_ecart_negatif", T.ByteType(), True),
        T.StructField("recours", T.StringType(), True),
    ]
)
siret_to_siren = sf_datalake.transform.SiretToSiren()
# Create a monthly date range that will become the time index
dr = pd.date_range(args.min_date, dt.date.today().isoformat(), freq="MS")
date_range = spark.createDataFrame(
    pd.DataFrame(dr.to_series().dt.date, columns=["période"])
)


debit = spark.read.csv(args.input, header=True, schema=debit_schema)
debit = siret_to_siren.transform(debit)

# Only select data located after the newly created time index: for a given "période"
# timestamp in the output dataframe, we only want to select debt data that concern past
# events, i.e. data located before "période."
debit = debit.select(
    [
        "siren",
        "siret",
        "date_traitement",
        "période_cotisation",
        "numéro_compte",
        "numéro_écart_négatif",
        "numéro_historique_écart_négatif",
        "dette_sociale_ouvrière",
        "dette_sociale_patronale",
    ]
).join(date_range, on=date_range["période"] >= debit["date_traitement"], how="inner")

# Each debt file, at URSSAF, has a "numéro_compte" identifier. Within this debt file,
# there may be sub-files, describing a precise due amount, indexed by
# "numéro_écart_négatif". This is the lower level of independent data that we'll
# consider. These indexes are defined within a given "période_cotisation" time frame.
w = (
    Window()
    .partitionBy(
        ["numéro_compte", "numéro_écart_négatif", "période", "période_cotisation"]
    )
    .orderBy(F.col("numéro_historique_écart_négatif").asc())
    .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
)

# We take the last value of contribution an each debt variable, as ordered by
# "numéro_historique_écart_négatif", which precisely indicates the last known value up
# to date for each "numéro_écart_négatif"-indexed debt. It is renamed to reflect this.
debit_par_compte = debit.select(
    [
        "siren",
        "période",
        "période_cotisation",
        "numéro_historique_écart_négatif",
        F.last("dette_sociale_ouvrière").over(w).alias("dette_sociale_ouvrière"),
        F.last("dette_sociale_patronale").over(w).alias("dette_sociale_patronale"),
        F.last("numéro_historique_écart_négatif")
        .over(w)
        .alias("indicateur_dernier_traitement"),
    ]
).filter(
    F.col("numéro_historique_écart_négatif") == F.col("indicateur_dernier_traitement")
)

# Handle missing values, sum by SIREN and export
mvh = sf_datalake.transform.MissingValuesHandler(
    inputCols=["dette_sociale_ouvrière", "dette_sociale_patronale"],
    value=configuration.preprocessing.fill_default_values,
)

summed_ds = debit_par_compte.groupby(["siren", "période"]).agg(
    F.sum("dette_sociale_ouvrière").alias("dette_sociale_ouvrière"),
    F.sum("dette_sociale_patronale").alias("dette_sociale_patronale"),
)

sf_datalake.io.write_data(mvh.transform(summed_ds), args.output, args.output_format)
