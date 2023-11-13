"""DGEFP historicized data preprocessing.

This script parses and extracts data from raw historicized files supplied by DGEFP. It
will output a dataset containing the following information:
- siren
- periode (date, first day of each month where data is available)
- apart_heures_consommees
- apart_heures_autorisees


"""

import datetime
import os
import sys
from os import path
from typing import List

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import PipelineModel, Transformer
from pyspark.sql.window import Window

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.configuration
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils


# Create a UDF to generate a list of dates within a date range
def generate_date_range(start, end):
    """
    Generates a list of dates within a given date range.

    Args :
    - start (datetime.date): The start date of the range.
    - end (datetime.date): The end date of the range.

    Returns:
    list of datetime.date: A list containing all dates within the specified range,
                           including both the start and end dates.
    """
    days = [(start + datetime.timedelta(days=i)) for i in range((end - start).days + 1)]
    return days


spark = sf_datalake.utils.get_spark_session()

parser = sf_datalake.io.data_path_parser()
parser.description = "Extract and pre-process DGEFP data."

parser.add_argument(
    "--configuration",
    help="""
    Configuration file name (including '.json' extension). If not provided,
    'standard.json' will be used.
    """,
    default="standard.json",
)

parser.add_argument(
    "--demande",
    dest="demande_data",
    help="Path to the 'demande' dataset.",
    required=True,
)
parser.add_argument(
    "--consommation",
    dest="consommation_data",
    help="Path to the 'consommation' dataset.",
    required=True,
)
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

# Parse configuration files and possibly override parameters.

config_file: str = args.pop("configuration")

configuration = sf_datalake.configuration.ConfigurationHelper(
    config_file=config_file, cli_args=args
)

# Load Data
consommation_schema = T.StructType(
    [
        T.StructField("id_da", T.StringType(), False),
        T.StructField("ETAB_SIRET", T.StringType(), False),
        T.StructField("heures", T.DoubleType(), True),
        T.StructField("montants", T.DoubleType(), True),
        T.StructField("effectifs", T.DoubleType(), True),
        T.StructField("mois", T.TimestampType(), True),
    ]
)
demande_schema = T.StructType(
    [
        T.StructField("ID_DA", T.StringType(), False),
        T.StructField("ETAB_SIRET", T.StringType(), False),
        T.StructField("EFF_ENT", T.DoubleType(), True),
        T.StructField("EFF_ETAB", T.DoubleType(), True),
        T.StructField("DATE_STATUT", T.TimestampType(), True),
        T.StructField("DATE_DEB", T.TimestampType(), True),
        T.StructField("DATE_FIN", T.TimestampType(), True),
        T.StructField("HTA", T.DoubleType(), True),
        T.StructField("MTA", T.DoubleType(), True),
        T.StructField("EFF_AUTO", T.DoubleType(), True),
        T.StructField("MOTIF_RECOURS_SE", T.IntegerType(), True),
        T.StructField("PERIMETRE_AP", T.IntegerType(), True),
        T.StructField("S_HEURE_CONSOM_TOT", T.DoubleType(), True),
        T.StructField("S_EFF_CONSOM_TOT", T.DoubleType(), True),
        T.StructField("S_MONTANT_CONSOM_TOT", T.DoubleType(), True),
        T.StructField("RECOURS_ANTERIEUR", T.IntegerType(), True),
    ]
)
demande = spark.read.csv(args.demande_data, header=True, schema=demande_schema)
consommation = spark.read.csv(
    args.demande_data, header=True, schema=consommation_schema
)

# Select ap spent and demand hours columns

demande = demande.select(["ETAB_SIRET", "DATE_STATUT", "DATE_DEB", "DATE_FIN", "HTA"])
consommation = consommation.select(["ETAB_SIRET", "mois", "heures"])

# Extract Siren from Siret
siretsiren_transformer = sf_datalake.transform.SiretToSiren(inputCol="ETAB_SIRET")
demande = siretsiren_transformer.transform(demande)
consommation = siretsiren_transformer.transform(consommation)

# Preprocess 'demande' dataset by aggreging data 'HTA'
# according to siren and timeframes (DATE_DEB / DATE_FIN)

generate_date_range_udf = F.udf(generate_date_range, T.ArrayType(T.DateType()))

# Add a new column with the list of dates in the range
demande = demande.withColumn(
    "DateRange", generate_date_range_udf(demande["DATE_DEB"], demande["DATE_FIN"])
)

# Explode the DateRange column to create one row per date
demande = demande.select("siren", "HTA", F.explode("DateRange").alias("Date"))

# Group by Id and Date, and sum the values
demande = demande.groupBy("siren", "Date").agg(F.sum("HTA").alias("Total_HTA"))

# Sort the DataFrame by Id and Date
demande = demande.orderBy("siren", "Date")

# Create a lag column to compare values with the previous row
window_spec = Window.partitionBy("siren").orderBy("Date")
demande = demande.withColumn("Previous_HTA", F.lag("Total_HTA").over(window_spec))

# Find the rows where the value changes
demande = demande.withColumn(
    "HTA_Change", F.when(F.col("Total_HTA") != F.col("Previous_HTA"), 1).otherwise(0)
)

# Create a cumulative sum of the Value_Change column
demande = demande.withColumn(
    "HTA_Change_Cumulative", F.sum("HTA_Change").over(window_spec)
)

# Group by Id and the cumulative sum of Value_Change,
# and find the minimum and maximum date in each group
demande = demande.groupBy("siren", "HTA_Change_Cumulative").agg(
    F.min("Date").alias("DATE_DEB_MR"),
    F.max("Date").alias("DATE_FIN_MR"),
    F.first("Total_HTA").alias("HTA"),
)

# Preprocess 'consommation' dataset by aggreging data 'heures'

sirenagg_transformer = sf_datalake.transform.SirenAggregator(
    grouping_cols=["siren", "mois"],
    aggregation_map={"heures": "sum"},
    no_aggregation=[],
)

consommation_preprocess = sirenagg_transformer.transform(consommation).drop(
    "ETAB_SIRET"
)

# Join 'demande' & 'consommation' dataset
ap_ds = (
    demande.join(
        consommation,
        on=(
            (consommation.siren == demande.siren)
            & (consommation.mois >= demande.DATE_DEB_MR)
            & (consommation.mois < demande.DATE_FIN_MR)
        ),
        how="inner",
    )
    .drop(demande.siren)
    .drop(demande.ETAB_SIRET)
)


ap_ds = ap_ds.select(
    F.col("siren"),
    F.col("mois").alias("periode"),
    F.col("heures").alias("apart_heures_consommees"),
    F.col("HTA").alias("apart_heures_autorisees"),
)

missingvalueHander_transformer = sf_datalake.transform.MissingValuesHandler(
    inputCols=["apart_heures_consommees", "apart_heures_autorisees"],
    value=configuration.preprocessing.fill_default_values,
)

complete_ad_ds = missingvalueHander_transformer.transform(ap_ds)

# calculate aggregates (MovingAverage & Diff)

n_months = 12
features = ["apart_heures_consommees", "apart_heures_autorisees"]
time_computations: List[Transformer] = []
for feature in features:
    time_computations.append(
        sf_datalake.transform.MovingAverage(inputCol=feature, n_months=n_months)
    )

time_computations.append(
    sf_datalake.transform.DiffOperator(
        inputCol="apart_heures_consommees", n_months=n_months, bfill=True
    )
)

output_ds = PipelineModel(stages=time_computations).transform(complete_ad_ds)


# Export output data


sf_datalake.io.write_data(output_ds, args.output, args.output_format)
