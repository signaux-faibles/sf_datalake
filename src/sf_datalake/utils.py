"""Utility functions."""

from typing import List

import pyspark.sql
import pyspark.sql.types as T
from pyspark.sql import SparkSession


def get_spark_session():
    """Creates or gets a SparkSession object."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.shuffle.blockTransferService", "nio")
    spark.conf.set("spark.driver.maxResultSize", "1300M")
    return spark


def numerical_columns(df: pyspark.sql.DataFrame) -> List[str]:
    """Returns a DataFrame's numerical data column names.

    Args:
        df: The input DataFrame.

    Returns:
        A list of column names.

    """
    numerical_types = (
        T.ByteType,
        T.DecimalType,
        T.DoubleType,
        T.FloatType,
        T.IntegerType,
        T.LongType,
        T.ShortType,
    )
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numerical_types)
    ]


def feature_index(config: dict) -> List[str]:
    """Generates an index associated with the features matrix columns.

    This index is used to keep track of the position of each features, which comes in
    handy in the explanation stage.

    Args:
        config: model configuration, as loaded by io.load_parameters().

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
