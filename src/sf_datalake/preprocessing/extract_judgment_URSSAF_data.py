"""Extract URSSAF judgment data for machine learning classification target definition.

The script expects the "siret", "date_effet", "action_procol" columns
inside the input table.

USAGE python extract_judgment_URSSAF_data.py <input_file> <output_file>
    [--start_date START_DATE] [--end_date END_DATE]

"""
import datetime
import os
import sys
from os import path

import pyspark.sql.functions as F
from pyspark.sql.window import Window

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
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()


df = spark.read.csv(
    args.input,
    inferSchema=True,
    header=True,
)

# Filter to restricted subset of input time period.
df = df.filter(
    (F.col("date_effet") >= args.start_date) & (F.col("date_effet") <= args.end_date)
)
# Get siren from siret
siren_siret_transformer = sf_datalake.transform.SiretToSiren(inputCol="siret")
df = siren_siret_transformer.transform(df)

# Get first judgment within input time period and only keep this judgment.
window_spec = Window.partitionBy("siren")
df_first_judg_date = (
    df.withColumn("date_jugement", F.min("date_effet").over(window_spec))
    .filter(F.col("date_effet") == F.col("date_jugement"))
    .select("siren", "date_jugement", F.col("action_procol").alias("nature_jugement"))
)

# Write output
sf_datalake.io.write_data(df_first_judg_date, args.output, args.output_format)
