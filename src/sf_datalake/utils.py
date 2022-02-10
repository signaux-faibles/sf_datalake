"""Utility functions."""

import json
from typing import Dict, List

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


def transformer_features_mapping(config: dict) -> Dict[str, List[str]]:
    """Associates each transformer with a list of features.

    Args:
        config: model configuration, as loaded by get_config().

    Returns:
        The transformer -> features mapping.

    """
    transformer_features = {}
    for feature, transformer in config["FEATURES"].items():
        transformer_features.setdefault(transformer, []).append(feature)
    return transformer_features
