"""DGEFP historicized data preprocessing.

This script parses and extracts data from raw historicized files supplied by DGEFP. It
will output a dataset containing the following information:
- siren
- periode (date, first day of each month where data is available)
- ap_consommation
- ap_demande


"""

import datetime
import os
import sys
from os import path

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
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

spark = sf_datalake.utils.get_spark_session()

parser = sf_datalake.io.data_path_parser()
parser.description = "Extract and pre-process DGEFP data."
parser.add_argument("--min_date", default="2014-01-01")
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

configuration = sf_datalake.configuration.ConfigurationHelper(config_file=config_file)

# Load Data
consommation_schema = T.StructType(
    [
        T.StructField("id_da", T.StringType(), False),
        T.StructField("siret", T.StringType(), False),
        T.StructField("ap_consommation", T.DoubleType(), True),
        T.StructField("montants", T.DoubleType(), True),
        T.StructField("effectifs", T.DoubleType(), True),
        T.StructField("periode", T.TimestampType(), True),
    ]
)
demande_schema = T.StructType(
    [
        T.StructField("id_da", T.StringType(), False),
        T.StructField("siret", T.StringType(), False),
        T.StructField("eff_ent", T.DoubleType(), True),
        T.StructField("eff_etab", T.DoubleType(), True),
        T.StructField("date_statut", T.TimestampType(), True),
        T.StructField("date_début", T.TimestampType(), True),
        T.StructField("date_fin", T.TimestampType(), True),
        T.StructField("ap_demande", T.DoubleType(), True),
        T.StructField("mta", T.DoubleType(), True),
        T.StructField("eff_auto", T.DoubleType(), True),
        T.StructField("motif_recours_se", T.IntegerType(), True),
        T.StructField("perimetre_ap", T.IntegerType(), True),
        T.StructField("s_heure_consom_tot", T.DoubleType(), True),
        T.StructField("s_eff_consom_tot", T.DoubleType(), True),
        T.StructField("s_montant_consom_tot", T.DoubleType(), True),
        T.StructField("recours_anterieur", T.IntegerType(), True),
    ]
)
demande = spark.read.csv(args.demande_data, header=True, schema=demande_schema)
consommation = spark.read.csv(
    args.demande_data, header=True, schema=consommation_schema
)

# Select ap spent and demand hours columns

demande = demande.select(
    ["siret", "date_statut", "date_début", "date_fin", "ap_demande"]
)
consommation = consommation.select(["siret", "periode", "ap_consommation"])

# Extract Siren from Siret
siretsiren_transformer = sf_datalake.transform.SiretToSiren(inputCol="siret")
demande = siretsiren_transformer.transform(demande)
consommation = siretsiren_transformer.transform(consommation)

# Preprocess 'demande' dataset by aggreging data 'ap_demande'
# according to siren and timeframes (date_début / date_fin)

# Indexing "demande dataframe" in order to indentify the demande
demande = demande.withColumn("siren_date_index", F.monotonically_increasing_id())

# Add a new column with the list of dates in the range


date_range = spark.createDataFrame(
    pd.date_range(args.min_date, datetime.date.today().isoformat()).to_frame(
        name="date"
    )
)
demande = demande.join(
    date_range,
    (date_range["date"] >= demande["date_début"])
    & (date_range["date"] <= demande["date_fin"]),
)

# Normalize the "Value" column (/days)

demande = demande.withColumn(
    "time_frame_size", F.expr("DATEDIFF(date_fin, date_début) + 1")
)
demande = demande.withColumn(
    "ap_demande", F.col("ap_demande") / F.col("time_frame_size")
)

# Group by siren and date, and sum the values for demand.
# Summing the siren_date_index aims to create a new id
# for intersection between two demands.
demande = demande.groupBy("siren", "date").agg(
    F.sum("ap_demande").alias("Total_ap_demande"),
    F.sum("siren_date_index").alias("Total_siren_date_index"),
)

# Next part of the code is used to merge intersection between demand time frames.
# Therefore, the goal is to compute new time frames based on intersection.
# Create a lag column to compare Total_siren_date_index with the previous row
window_spec = Window.partitionBy("siren").orderBy("date")
demande = demande.withColumn(
    "Previous_siren_date_index",
    F.lag("Total_siren_date_index").over(window_spec),
)

# Find the rows where the siren_date_index changes
demande = demande.withColumn(
    "siren_date_index_Change",
    F.when(
        F.col("Total_siren_date_index") != F.col("Previous_siren_date_index"),
        1,
    ).otherwise(0),
)

# Create groups based on cumulative sum of the siren_date_index_Change column
demande = demande.withColumn(
    "group", F.sum("siren_date_index_Change").over(window_spec)
)

# Group by siren and group to create new distinct timeframes and,
# compute de total amount for "ap_demande".
# The value of ap_demande is the same in a group.
demande = demande.groupBy("siren", "group").agg(
    F.min("date").alias("date_début_mr"),
    F.max("date").alias("date_fin_mr"),
    F.first("Total_ap_demande").alias("ap_demande"),
)


# Preprocess 'consommation' dataset by aggreging data 'ap_consommation'

sirenagg_transformer = sf_datalake.transform.SirenAggregator(
    grouping_cols=["siren", "periode"],
    aggregation_map={"ap_consommation": "sum"},
    no_aggregation=[],
)

consommation_preprocess = sirenagg_transformer.transform(consommation).drop("siret")

# Join 'demande' & 'consommation' dataset
ap_ds = demande.join(
    consommation,
    on=(
        (consommation.siren == demande.siren)
        & (consommation.periode >= demande.date_début_mr)
        & (consommation.periode < demande.date_fin_mr)
    ),
    how="inner",
).drop(demande.siren)


ap_ds = ap_ds.select(
    F.col("siren"),
    F.col("periode").alias("periode"),
    F.col("ap_consommation"),
    F.col("ap_demande"),
)

# Manage missing values
missingvalueHander_transformer = sf_datalake.transform.MissingValuesHandler(
    inputCols=["ap_consommation", "ap_demande"],
    value=configuration.preprocessing.fill_default_values,
)

output_ds = missingvalueHander_transformer.transform(ap_ds)

# Export output data

sf_datalake.io.write_data(output_ds, args.output, args.output_format)
