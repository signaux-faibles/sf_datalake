"""Utility functions."""

import datetime as dt
from typing import List

import pyspark.sql
import pyspark.sql.functions as F
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


def extract_column_names(df: pyspark.sql.DataFrame, assembled_column: str) -> List[str]:
    """Get inner column names from an assembled column.

    Here, "assembled" means: that has been transformed using a VectorAssembler.

    Args:
        df: A DataFrame
        assembled_column: The "assembled" column name.

    Returns:
        A list of column names.

    """
    column_metadata = df.schema[assembled_column].metadata["ml_attr"]
    columns = [None] * column_metadata["num_attrs"]
    for _, variables in column_metadata["attrs"].items():
        for variable_dict in variables:
            columns[variable_dict["idx"]] = variable_dict["name"]
    return columns


def to_date(str_date: str, date_format="%Y-%m-%d") -> dt.date:
    """Convert string date to datetime.date object"""
    return dt.datetime.strptime(str_date, date_format).date()


def count_nan_values(
    df: pyspark.sql.DataFrame,
) -> pyspark.sql.Row:
    """Counts number of NaN values in numerical columns.

    Args:
        df: The input DataFrame.

    Returns:
        A Row specifying the number of NaN values in numerical fields.

    """
    return df.select(
        [F.count(F.when(F.isnull(c), c)).alias(c) for c in numerical_columns(df)]
    ).collect()[0]


def count_missing_values(df: pyspark.sql.DataFrame) -> pyspark.sql.Row:
    """Counts number of null values in each column.

    Args:
        df: The input DataFrame.

    Returns:
        A Row specifying the number of null values for each column.

    """
    return df.select(
        [F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns]
    ).collect()[0]
