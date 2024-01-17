"""DARES historicized "activité partielle" data pre-processing.

This script parses and extracts data from raw historicized files supplied by DGEFP. It
will output a dataset containing the following information:
- siren.
- période: date, first day of each month where data is available.
- ap_heures_consommées: number of 'activité partielle' (partial unemployment) hours used
  over a given (siren, période) couple.
- ap_heures_autorisées: granted 'activité partielle' at a given `(siren, période)`
  couple. This data's unit is a number of hours, normalized by the number of days
  contained in the time frame over which the request was made. If multiple requests were
  made for a given `période`, the normalized data is summed over all found requests.

"""

import argparse
import os
import sys
from os import path

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import PipelineModel
from pyspark.sql.window import Window

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))

# pylint: disable=C0413
import pandas as pd

# isort: on
import sf_datalake.configuration
import sf_datalake.io
import sf_datalake.transform
import sf_datalake.utils

spark = sf_datalake.utils.get_spark_session()

parser = argparse.ArgumentParser(description="Extract and pre-process DGEFP data.")
parser.add_argument(
    "--min_date",
    default="2014-01-01",
    help="Minimum date to consider for requested ap data.",
)
parser.add_argument(
    "--max_date",
    default="2100-01-01",
    help="Maximum date to consider for requested ap data.",
)
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
parser.add_argument("--output", help="The output path.")
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

# Parse configuration files and possibly override parameters.

configuration = sf_datalake.configuration.ConfigurationHelper(
    config_file=args.configuration
)

# Load Data
consommation_schema = T.StructType(
    [
        T.StructField("id_da", T.StringType(), True),
        T.StructField("siret", T.StringType(), False),
        T.StructField("ap_heures_consommées", T.DoubleType(), True),
        T.StructField("montants", T.DoubleType(), True),
        T.StructField("effectifs", T.DoubleType(), True),
        T.StructField("période", T.DateType(), False),
    ]
)
demande_schema = T.StructType(
    [
        T.StructField("id_da", T.StringType(), True),
        T.StructField("siret", T.StringType(), False),
        T.StructField("eff_ent", T.DoubleType(), True),
        T.StructField("eff_étab", T.DoubleType(), True),
        T.StructField("date_statut", T.DateType(), True),
        T.StructField("date_début", T.DateType(), True),
        T.StructField("date_fin", T.DateType(), True),
        T.StructField("hta", T.DoubleType(), True),
        T.StructField("mta", T.DoubleType(), True),
        T.StructField("eff_auto", T.DoubleType(), True),
        T.StructField("motif_recours_se", T.IntegerType(), True),
        T.StructField("périmètre_ap", T.IntegerType(), True),
        T.StructField("s_heure_consom_tot", T.DoubleType(), True),
        T.StructField("s_eff_consom_tot", T.DoubleType(), True),
        T.StructField("s_montant_consom_tot", T.DoubleType(), True),
        T.StructField("recours_antérieur", T.IntegerType(), True),
    ]
)

# Select required columns and filter "demande" set according to the reason the
# unemployment authorization was requested.
demande = spark.read.csv(args.demande_data, header=True, schema=demande_schema).filter(
    F.col("motif_recours_se") < 6
)
consommation = spark.read.csv(
    args.consommation_data, header=True, schema=consommation_schema
)
demande = demande.select(["siret", "date_statut", "date_début", "date_fin", "hta"])
consommation = consommation.select(["siret", "période", "ap_heures_consommées"])

siret_to_siren_transformer = sf_datalake.transform.SiretToSiren(inputCol="siret")

### "Demande" dataset
# Create the time index for the output DataFrame. For now it has daily frequency in
# order to normalize and aggregate data easily.
date_range = spark.createDataFrame(
    pd.DataFrame(
        pd.date_range(args.min_date, args.max_date, freq="D").to_series(),
        columns=["période"],
    )
)

demande = demande.join(
    date_range,
    (date_range["période"] >= demande["date_début"])
    & (date_range["période"] <= demande["date_fin"]),
)


# Normalize by timeframes length, in days.
demande = demande.withColumn(
    "ap_heures_autorisées_par_jour",
    F.col("hta") / (F.datediff(end="date_fin", start="date_début") + 1),
)

# We determine disjoint time intervals for a given SIRET and merge them as follows:
# - For each new start date that appears, we check if it is located later in time than
#   the latest end date associated with the previous start date. We keep track of where
#   these changes occur.
# - We add up every granted "ap" (per unit of time) located between the tracked changes
#   and create new intervals boundaries that cover every previously intersecting
#   timeframes.
w = (
    Window.partitionBy("siret")
    .orderBy("date_début")
    .rangeBetween(Window.unboundedPreceding, Window.currentRow)
)

demande = (
    demande.withColumn("date_fin_max_cumulé", F.max("date_fin").over(w))
    .withColumn(
        "nouvel_intervalle",
        F.when(
            F.col("date_début")
            > F.lag("date_fin_max_cumulé").over(
                Window.partitionBy("siret").orderBy("date_début")
            ),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    # The cumulative sum uniquely indentifies a new disjoint interval.
    .withColumn("id_intervalle", F.sum("nouvel_intervalle").over(w))
    .drop("nouvel_intervalle", "date_fin_max_cumulé")
)

# Sum over newly defined merged timeframes, then over siren. The first aggregation is
# done over all days belonging to the same month so that "période" becomes a monthly
# index.
demande_agg = (
    siret_to_siren_transformer.transform(
        demande.groupBy(
            [
                F.date_trunc(format="month", timestamp="période")
                .cast("date")
                .alias("période"),
                "siret",
                "id_intervalle",
            ]
        ).agg(
            F.sum("ap_heures_autorisées_par_jour").alias("ap_heures_autorisées"),
            # TODO: we may want to keep these boundary dates by early exporting
            # SIRET-level data here
            F.min("date_début").alias("ap_date_début_autorisation"),
            F.max("date_fin").alias("ap_date_fin_autorisation"),
        )
    )
    .groupBy(["siren", "période"])
    .agg(
        F.sum("ap_heures_autorisées").alias("ap_heures_autorisées"),
    )
)

# Restrict dataset to user-input dates
demande_out = demande_agg.filter(F.col("période").between(args.min_date, args.max_date))

### 'consommation' dataset
siren_aggregator = sf_datalake.transform.SirenAggregator(
    grouping_cols=["siren", "période"],
    aggregation_map={"ap_heures_consommées": "sum"},
    no_aggregation=[],
)
consommation_pipeline_model = PipelineModel(
    [siret_to_siren_transformer, siren_aggregator]
)

consommation_out = consommation_pipeline_model.transform(consommation)

# Join 'demande' & 'consommation' dataset
ap_ds = demande_out.join(
    consommation_out,
    on=["période", "siren"],
    how="outer",
).select("siren", "période", "ap_heures_consommées", "ap_heures_autorisées")

### Manage missing values and export
output_ds = sf_datalake.transform.MissingValuesHandler(
    inputCols=["ap_heures_consommées", "ap_heures_autorisées"],
    value=configuration.preprocessing.fill_default_values,
).transform(ap_ds)

sf_datalake.io.write_data(output_ds, args.output, args.output_format)
