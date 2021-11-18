"""Utility functions for data handling.

"""
from os import path

from pyspark.sql import DataFrame, SparkSession  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401
from pyspark.sql.types import StringType  # pylint: disable=E0401


def instantiate_spark_session():
    """Creates or gets a SparkSession object."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.shuffle.blockTransferService", "nio")
    spark.conf.set("spark.driver.maxResultSize", "1300M")
    return spark


def load_source(src_path: str, spl_size: int = None) -> DataFrame:
    """Loads some orc-stored data."""
    spark = instantiate_spark_session()
    df = spark.read.orc(src_path)
    if spl_size is not None:
        df = df.sample(spl_size)
    return df


def csv_to_orc(input_filename: str, output_filename: str):
    """Writes a file stored as csv in orc format."""
    spark = instantiate_spark_session()
    df = spark.read.options(inferSchema="True", header="True", delimiter="|").csv(
        path.join(input_filename)
    )
    df.write.format("orc").save(output_filename)


def stringify_and_pad_siren(input_path: str, output_path: str):
    """Reads an orc file and normalizes its "siren" entries."""
    spark = instantiate_spark_session()
    df = spark.read.orc(input_path)
    df = df.withColumn("siren", F.col("siren").cast(StringType()))
    df = df.withColumn("siren", F.lpad(df["siren"], 9, "0"))
    df.write.format("orc").save(output_path)
