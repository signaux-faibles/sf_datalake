"""Utility functions."""

import datetime as dt
from typing import List

import pyspark.sql
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window as W
from pyspark.sql import functions as F


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


def merge_asof(  # pylint: disable=too-many-locals, too-many-arguments
    df_left: pyspark.sql.DataFrame,
    df_right: pyspark.sql.DataFrame,
    on: str,
    by: str = None,
    tolerance: int = None,
    direction: str = "backward",
) -> pyspark.sql.DataFrame:
    """Perform a merge by key distance.

    This is similar to a left-join except that we match on nearest key
    rather than equal keys. Both DataFrames must be sorted by the
    key. This key should be of date type.

    This function performs an asof merge on two DataFrames based on a
    specified column 'on'. It supports grouping by additional columns
    specified in 'by'. The 'tolerance' parameter allows for merging
    within a specified difference range. The 'direction' parameter
    determines the direction of the asof merge.

    Args:
        df_left : The left DataFrame to be merged.
        df_right : The right DataFrame to be merged.
        on : The column on which to merge the DataFrames.
        by (optional): The column(s) to group by before merging.
        tolerance (optional): The maximum difference allowed for asof merging, in
          days.
        direction : The direction of asof merging ('backward', 'forward', or 'nearest').

    Returns:
        DataFrame resulting from the asof merge.
    """

    def backward(w, stru, vc):
        return add_diff(F.last(stru, True).over(w), vc)

    def forward(w, stru, vc):
        return add_diff(
            F.first(stru, True).over(w.rowsBetween(0, W.unboundedFollowing)), vc
        )

    def nearest(w, stru, vc):
        return F.sort_array(
            F.array(backward(w, stru, vc), forward(w, stru, vc))
        ).getItem(0)

    def add_diff(struct_col, value_col):
        """
        Adds a 'diff' column to the input struct 'struct_col' representing the absolute
        difference in days between the 'on' column and each element in 'struct_col[on]'.

        Parameters:
        - struct_col : The input struct column containing
                        the 'on' and 'value_col' fields.
        - value_col : The name of the column representing values in the input struct.

        Returns:
         A struct column with three fields: 'diff', 'on', and 'value_col'.
        """
        return F.struct(
            F.abs(F.datediff(F.col(on), struct_col[on])).alias("diff"),
            struct_col[on].alias(on),
            struct_col[value_col].alias(value_col),
        )

    # Handle cases where 'by' is not specified
    df_r = df_right if by else df_right.withColumn("_by", F.lit(1))
    df_l = df_left if by else df_left.withColumn("_by", F.lit(1))
    df_l = df_l.withColumn("_df_l", F.lit(True))
    if by is None:
        by = ["_by"]
    # In other cases, use specified group key(s)
    elif isinstance(by, str):
        by = [by]
    else:
        assert all(isinstance(s, str) for s in by), TypeError(
            "All elements of `by` should by strings"
        )

    join_on = [on] + by
    df = df_l.join(df_r, join_on, "full")
    w0 = W.partitionBy(*by).orderBy(on)

    # TODO: this is where we explain what happens in this loop
    for c in set(df_right.columns) - set(join_on):
        stru1 = F.when(~F.isnull(c), F.struct(on, c))
        window_function = {
            "backward": backward,
            "forward": forward,
            "nearest": nearest,
        }
        stru2 = window_function[direction](w0, stru1, c)
        if tolerance:
            diff_col = F.abs(F.datediff(F.col(on), stru2[on])).alias("diff")
            c_col = F.when(diff_col <= tolerance, stru2[c]).otherwise(F.col(c))
            df = df.withColumn(c, c_col)
        else:
            # If no tolerance specified, directly use the result of stru2
            df = df.withColumn(c, stru2[c])

    # Filter as if we'd done a left join, drop temporary columns.
    return df.filter("_df_l").drop("_df_l", "_by")
