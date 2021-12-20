"""Utility functions.

"""
from pyspark.sql import SparkSession  # pylint: disable=E0401


def instantiate_spark_session():
    """Creates or gets a SparkSession object."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.shuffle.blockTransferService", "nio")
    spark.conf.set("spark.driver.maxResultSize", "1300M")
    return spark
