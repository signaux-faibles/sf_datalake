"""Utility functions."""

import datetime as dt
import functools
import operator
from typing import Any, List, Union

import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window as W


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


def clip(
    col: Union[str, pyspark.sql.Column], lower: Any = None, upper: Any = None
) -> pyspark.sql.Column:
    """Trim values at input threshold(s).

    Args:
        col: A column object or a column name.
        lower: Minimum value to clip.
        upper: Minimum value to clip.

    Returns:
        A colum clipped between the input  min and/or max values.
    """
    if isinstance(col, pyspark.sql.Column):
        trimmed_col = col
    elif isinstance(col, str):
        trimmed_col = F.col(col)
    else:
        raise ValueError(f"col should be a Column object or str, but got {type(col)}.")
    if lower:
        trimmed_col = F.when(trimmed_col < lower, lower).otherwise(trimmed_col)
    if upper:
        trimmed_col = F.when(trimmed_col > upper, upper).otherwise(trimmed_col)
    return trimmed_col


def merge_asof(  # pylint: disable=too-many-arguments, too-many-locals
    df_left: pyspark.sql.DataFrame,
    df_right: pyspark.sql.DataFrame,
    on: str = "période",
    by: Union[str, List] = "siren",
    tolerance: int = None,
    direction: str = "backward",
) -> pyspark.sql.DataFrame:
    """Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest key rather than equal
    keys. The associated column should be of date type.

    This function performs an asof merge on two DataFrames based on a specified column
    'on'. It supports grouping by additional columns specified in 'by'. The 'tolerance'
    parameter allows for merging within a specified difference range. The 'direction'
    parameter determines the direction of the asof merge.

    Args:
        df_left : The left DataFrame to be merged.
        df_right : The right DataFrame to be merged.
        on : The date column on which to merge the DataFrames.
        by (optional): The column(s) to group by before merging.
        tolerance (optional): The maximum difference allowed for asof merging, in
          days.
        direction : The direction of asof merging ('backward', 'forward', or 'nearest').

    Returns:
        DataFrame resulting from the asof merge.

    """

    def backward(
        w: pyspark.sql.WindowSpec, col: Union[pyspark.sql.column.Column, str]
    ) -> pyspark.sql.column.Column:
        return F.last(col, ignorenulls=True).over(
            w.rowsBetween(W.unboundedPreceding, W.currentRow)
        )

    def forward(
        w: pyspark.sql.WindowSpec,
        col: Union[pyspark.sql.column.Column, str],
    ) -> pyspark.sql.column.Column:
        return F.first(col, ignorenulls=True).over(
            w.rowsBetween(W.currentRow, W.unboundedFollowing)
        )

    # Handle cases where 'by' is not specified
    df_r = df_right if by else df_right.withColumn("_by", F.lit(True))
    df_l = df_left if by else df_left.withColumn("_by", F.lit(True))
    if by is None:
        by = ["_by"]
    # In other cases, use specified group key(s)
    elif isinstance(by, str):
        by = [by]
    elif isinstance(by, list):
        assert all(isinstance(s, str) for s in by), TypeError(
            "All elements of `by` should by strings"
        )
    else:
        raise ValueError("`by` should either be None, str or list.")

    df_l = df_l.withColumn("from_df_l", F.lit(True))
    join_keys = by + [on]

    # Join using only specified `by` and `on` from right df, then create `coalesced_key`
    # columns in case there are any missing `on` values subsequent to the merge.
    df = df_l.alias("left").join(
        df_r.select(join_keys).alias("right_join_keys"),
        on=functools.reduce(
            operator.and_,
            (
                F.col(f"left.{key}") == F.col(f"right_join_keys.{key}")
                for key in join_keys
            ),
        ),
        how="outer",
    )
    for key in join_keys:
        df = df.withColumn(
            f"coalesced_{key}",
            F.coalesce(F.col(f"left.{key}"), F.col(f"right_join_keys.{key}")),
        )

    # Compute a target join date for each row, among available dates in df_right, using
    # a forward / backward time window.
    window_spec = W.partitionBy(*[f"coalesced_{key}" for key in by]).orderBy(
        F.col(f"coalesced_{on}")
    )
    window_function = {
        "backward": backward,
        "forward": forward,
    }
    df = df.withColumn(
        "target_join_date",
        window_function[direction](window_spec, f"right_join_keys.{on}"),
    )

    # Drop duplicate columns and replace with the coalesced ones now that we're done
    # with target date computing.
    df = df.select(
        *(f"left.{col}" for col in set(df_l.columns) - set(join_keys)),
        *(f"coalesced_{key}" for key in join_keys),
        "target_join_date",
    )
    for key in join_keys:
        df = df.withColumnRenamed(f"coalesced_{key}", key)

    # If the date difference between left and right dates is greater than tolerance, a
    # flag that will prevent the join is added.
    if tolerance:
        sign = 1 if direction == "forward" else -1
        df = df.withColumn(
            "_do_not_join",
            F.when(
                (sign * F.datediff(F.col("target_join_date"), F.col(f"{on}")))
                <= tolerance,
                False,
            ).otherwise(True),
        )
        by.append("_do_not_join")

    # Actually join right df data over `by` and pre-computed target date. Filter as if
    # we'd done a left join during the outer join, drop temporary columns.
    df = (
        df.filter("from_df_l")
        .alias("pre_join")
        .join(
            df_r.withColumn("_do_not_join", F.lit(False)).alias("right"),
            on=functools.reduce(
                operator.and_,
                (F.col(f"pre_join.{key}") == F.col(f"right.{key}") for key in by),
            )
            & (F.col("pre_join.target_join_date") == F.col(f"right.{on}")),
            how="left",
        )
    ).drop("from_df_l", "_by", "_do_not_join", "target_join_date")

    return df.select(
        "pre_join.*", *(f"right.{col}" for col in set(df_r.columns) - set(join_keys))
    )


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
