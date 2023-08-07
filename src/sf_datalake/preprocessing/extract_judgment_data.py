"""Extract judgment data for machine learning classification target definition.

The script expects the "siren", "djug", "najug" columns inside the input table. The
latter correspond to the date and type of judgment (represented using a one character
encoding)

USAGE python extract_judgment_data.py <input_file> <output_file>
    [--start_date START_DATE] [--end_date END_DATE]

"""
import datetime
import os
import sys
from os import path

import pyspark.sql.functions as F

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
parser.description = "Extract judgment data"
parser.add_argument("--start_date", type=str, default="2014-01-01")
parser.add_argument("--end_date", type=str, default=datetime.date.today().isoformat())
args = parser.parse_args()


df = spark.read.csv(
    args.INPUT_FILE,
    sep="|",
    inferSchema=True,
    header=True,
)
df = df.withColumn("djug", F.to_date(F.col("djug").cast("string"), "yyyyMMdd"))

judgment_codes = {
    "1": "LIQUIDATION DE BIENS",
    "2": "REGLEMENT JUDICIAIRE",
    "3": "REDRESSEMENT JUDICIAIRE",
    "4": "LIQUIDATION JUDICIAIRE",
    # "5": PLAN DE REDRESSEMENT,
    # "6": PLAN DE CESSION ,
    # "7": PLAN DE CONTINUATION,
    "8": "JUGEMENT DE SAUVEGARDE",
    # "A": PLAN DE SAUVEGARDE,
}

# Filter to restricted subset of judgment types and input time period.
df_judg = df.filter(
    (F.col("najug").isin(list(judgment_codes)))
    & (F.col("djug") >= args.start_date)
    & (F.col("djug") <= args.end_date)
)


# Get first judgment within input time period and only keep this judgment.
df_first_judg_date = df_judg.groupby("siren").agg(F.min("djug").alias("djug"))

# Write output
df_first_judg_date.withColumnRenamed("djug", "date_jugement").write.csv(
    args.output, header=True
)
