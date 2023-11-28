"""Extract URSSAF judgment data for machine learning classification target definition.

The script expects at least "siret", "date_effet" columns inside the input table.

USAGE python extract_judgment_URSSAF_data.py <input_file> <output_file>
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

# pylint: disable=duplicate-code
parser = sf_datalake.io.data_path_parser()
parser.description = "Extract judgment data from URSSAF source."
parser.add_argument("--start_date", type=str, default="2014-01-01")
parser.add_argument("--end_date", type=str, default=datetime.date.today().isoformat())
parser.add_argument(
    "--output_format", default="orc", help="Output dataset file format."
)
args = parser.parse_args()

# Filter to restricted input time period.
df = spark.read.csv(
    args.input,
    inferSchema=True,
    header=True,
)
df = df.filter(
    (F.col("date_effet") >= args.start_date) & (F.col("date_effet") <= args.end_date)
)
# Group by SIREN, then get first judgment within input time period
df = sf_datalake.transform.SiretToSiren(inputCol="siret").transform(df)
df_output = df.groupBy("siren").agg(F.min("date_effet").alias("date_jugement"))

# Write output
sf_datalake.io.write_data(df_output, args.output, args.output_format)
