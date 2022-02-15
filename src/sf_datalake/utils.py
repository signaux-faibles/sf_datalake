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


def feature_index(config: dict) -> List[str]:
    """Generates an index associated with the features matrix columns.

    This index is used to keep track of the position of each features, which comes in
    handy in the explanation stage.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A list of features ordered as they are inside the features matrix.

    """
    indexer: List[str] = []
    for transformer, features in config["TRANSFORMER_FEATURES"].items():
        if transformer == "StandardScaler":
            indexer.extend(features)
        elif transformer == "OneHotEncoder":
            for feature in features:
                indexer.extend(
                    [
                        f"{feature}_ohcat{i}"
                        for i, _ in enumerate(config["ONE_HOT_CATEGORIES"][feature])
                    ]
                )
        else:
            raise NotImplementedError(
                f"Indexing for transformer {transformer} is not implemented yet."
            )
    return indexer
