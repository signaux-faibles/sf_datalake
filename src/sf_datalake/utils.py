"""Utility functions."""

import json

import pkg_resources
from pyspark.sql import SparkSession


def get_spark_session():
    """Creates or gets a SparkSession object."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.shuffle.blockTransferService", "nio")
    spark.conf.set("spark.driver.maxResultSize", "1300M")
    return spark


def get_config(config_fname: str) -> dict:
    """Loads a model run config from a preset config json file.

    Args:
        config_name: Basename of a config file (including .json extension).

    Returns:
        The config parameters.

    """

    with pkg_resources.resource_stream("sf_datalake", f"config/{config_fname}") as f:
        config = json.load(f)
    return config
