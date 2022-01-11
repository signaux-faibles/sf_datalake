"""Utility functions."""

import json
from typing import List, Tuple

import pkg_resources
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType


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
    return config


def is_centered(df: pyspark.sql.DataFrame, tol: float) -> Tuple[bool, List]:
    """Check if a DataFrame has a `features` column with centered individual variables.
    `features` column is the result of at least a `VectorAssembler()`.

    Args:
        df : Input DataFrame.
        tol :  a tolerance for the zero equality test.

    Returns:
        Tuple[bool, List]: True if variables are centered else False. A list of the
                            mean of each variable.

    Example:
        is_centered(train_transformed.select(["features"]), tol = 1E-8)
    """
    assert "features" in df.columns, "Input DataFrame doesn't have a 'features' column."

    dense_to_array_udf = F.udf(lambda v: [float(x) for x in v], ArrayType(FloatType()))

    df = df.withColumn("features_array", dense_to_array_udf("features"))
    n_features = len(df.first()["features"])

    df_agg = df.agg(
        F.array(*[F.avg(F.col("features_array")[i]) for i in range(n_features)]).alias(
            "mean"
        )
    )
    all_col_means = df_agg.select(F.col("mean")).collect()[0]["mean"]

    return (all(x < tol for x in all_col_means), all_col_means)
