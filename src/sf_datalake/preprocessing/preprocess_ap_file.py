"""DGEFP historicized data preprocessing.

This script parses and extracts data from raw historicized files supplied by DGEFP. It
will output a dataset containing the following information:
- siren
- periode (date, first day of each month where data is available)
- apart_heures_consommees
- apart_heures_autorisees


"""

import os
import sys
from os import path
from typing import List
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel, Transformer
from pyspark.sql.types import LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, TimestampType, DoubleType, DateType
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

spark = sf_datalake.utils.get_spark_session()

parser = sf_datalake.io.data_path_parser()
parser.description = "Extract and pre-process DGEFP data."
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

# Load Data 
consommation_schema = siren_schema = StructType(
    [
        StructField("id_da", StringType(), False),
        StructField("ETAB_SIRET", StringType(), False),
        StructField("heures", DoubleType(), True),
        StructField("montants", DoubleType(), True),
        StructField("effectifs", DoubleType(), True),
        StructField("mois", TimestampType(), True),

    ]
)
demande_schema = siren_schema = StructType(
    [
        StructField("ID_DA", StringType(), False),
        StructField("ETAB_SIRET", StringType(), False),
        StructField("EFF_ENT", DoubleType(), True),
        StructField("EFF_ETAB", DoubleType(), True),
        StructField("DATE_STATUT", TimestampType(), True),
        StructField("DATE_DEB", TimestampType(), True),
        StructField("DATE_FIN", TimestampType(), True),
        StructField("HTA", DoubleType(), True),
        StructField("MTA", DoubleType(), True),
        StructField("EFF_AUTO", DoubleType(), True),
        StructField("MOTIF_RECOURS_SE", IntegerType(), True),
        StructField("PERIMETRE_AP", IntegerType(), True),
        StructField("S_HEURE_CONSOM_TOT", DoubleType(), True),
        StructField("S_EFF_CONSOM_TOT", DoubleType(), True),
        StructField("S_MONTANT_CONSOM_TOT", DoubleType(), True),
        StructField("RECOURS_ANTERIEUR", IntegerType(), True),
     


        
    ]
)
demande = spark.read.csv(args.demande_data, header=True, schema=demande_schema)
consommation = spark.read.csv(args.demande_data, header=True, schema=consommation_schema)

# Select ap mount columns 

demande = demande.select(['ETAB_SIRET', 'DATE_STATUT', 'DATE_DEB', 'DATE_FIN', 'HTA'])
consommation = consommation.select(['ETAB_SIRET', 'mois', 'heures'])

# SiretToSiren

SiretSirenPipeline = PipelineModel(
    [
        sf_datalake.transform.SiretToSiren(inputCol='ETAB_SIRET'),        
        
    ]
)
demande = SiretSirenPipeline.transform(demande)
consommation = SiretSirenPipeline.transform(consommation)

# Preprocess 'demande' dataset by aggreging data 'HTA' according to siren and timeframes (DATE_DEB / DATE_FIN)

# Create a UDF to generate a list of dates within a date range
def generate_date_range(start, end):
    days = [(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    return days

generate_date_range_udf = F.udf(generate_date_range, ArrayType(DateType()))

# Add a new column with the list of dates in the range
demande = demande.withColumn("DateRange", generate_date_range_udf(demande["DATE_DEB"], demande["DATE_FIN"]))

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
demande = demande.withColumn("HTA_Change", F.when(F.col("Total_HTA") != F.col("Previous_HTA"), 1).otherwise(0))

# Create a cumulative sum of the Value_Change column
demande = demande.withColumn("HTA_Change_Cumulative", F.sum("HTA_Change").over(window_spec))

# Group by Id and the cumulative sum of Value_Change, and find the minimum and maximum date in each group
demande = demande.groupBy("siren", "HTA_Change_Cumulative").agg(
    F.min("Date").alias("DATE_DEB_MR"),
    F.max("Date").alias("DATE_FIN_MR"),
    F.first("Total_HTA").alias("HTA")
)

# Preprocess 'consommation' dataset by aggreging data 'heures'

SirenAggPipeline = PipelineModel(
    [
        sf_datalake.transform.SirenAggregator(
            grouping_cols=['siren', 'mois'],
            aggregation_map= {'heures':"sum"},
            no_aggregation=[],
        )        
    ]
)
consommation_preprocess = SirenAggPipeline.transform(consommation).drop('ETAB_SIRET')

# Join 'demande' & 'consommation' dataset
output_ds = (    demande.join(
        consommation,
        on=(
            (consommation.siren == demande.siren)
            & (consommation.mois >= demande.DATE_DEB_MR)
            & (consommation.mois < demande.DATE_FIN_MR)
        ),
        how="inner",
    ).drop(demande.siren).drop(demande.ETAB_SIRET)
)

# Export output data

output_ds = output_ds.select(F.col("siren"), F.col("mois").alias("periode"), F.col("heures").alias("apart_heures_consommees"), F.col("HTA").alias("apart_heures_autorisees"))

sf_datalake.io.write_data(output_ds, args.output, args.output_format)
