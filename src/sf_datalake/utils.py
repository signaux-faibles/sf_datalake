"""Utility functions.

"""
import json

import pkg_resources
from pyspark.sql import SparkSession


def instantiate_spark_session():
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

    config["TAC_VARIABLES"] = {f"tac_1y_{v}" for v in config["MRV_VARIABLES"]}
    config["SF_VARIABLES"] = list(
        config["SUM_VARIABLES"] + config["AVG_VARIABLES"] + config["COMP_VARIABLES"]
    )
    config["FEATURES"] = list(set(config["SF_VARIABLES"] + config["MRV_VARIABLES"]))

    config["FEATURES_TO_STANDARDSCALER"] = config["FEATURES"]
    config["TRANSFORMERS"] = [(config["FEATURES_TO_STANDARDSCALER"], "StandardScaler")]
    return config
