"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import Iterable, List, Tuple

import pyspark
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType


def count_missing_values(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Counts number of null values in each column.

    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame specifying the number of null values for each column.

    """
    return df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])


def count_nan_values(
    df: pyspark.sql.DataFrame,
    omit_type: Iterable[str] = ("timestamp", "string", "date", "bool"),
) -> pyspark.sql.DataFrame:
    """Counts number of NaN values in numerical columns.

    Args:
        df: The input DataFrame.
        omit_type: An iterable containing the names of types that should not be looked
          for.

    Returns:
        A DataFrame specifying the number of NaN values in numerical fields.

    """
    return df.select(
        [
            F.count(F.when(F.isnan(c), c)).alias(c)
            for (c, c_type) in df.dtypes
            if c_type not in omit_type
        ]
    )


def accounting_time_span(
    df: pyspark.sql.DataFrame,
) -> Tuple[pyspark.sql.Row, pyspark.sql.Row]:
    """Computes global time spans of companies accouting years.

    Args:
        df: The input DataFrame. It must have a "date_deb_exercice" and
          "date_fin_exercice" columns.

    Returns:
        A couple of DataFrame objects, each containing the min and max dates associated
          with the beginning and end of accounting years ("exercice comptable").

    """
    assert {"date_deb_exercice", "date_fin_exercice"} <= set(df.columns)

    date_deb_exercice_span = df.select(
        F.min("date_deb_exercice"),
        F.max("date_deb_exercice"),
    ).first()
    date_fin_exercice_span = df.select(
        F.min("date_fin_exercice"),
        F.max("date_fin_exercice"),
    ).first()

    return date_deb_exercice_span, date_fin_exercice_span


def accounting_year_distribution(
    df: pyspark.sql.DataFrame, unit: str = "days", duration_col: str = "ay_duration"
) -> pyspark.sql.DataFrame:
    """Computes declared accounting year duration distribution.

    Args:
        df: The input DataFrame. It must have a "date_deb_exercice" and
          "date_fin_exercice" columns.
        unit: The unit measuring accounting year duration. Should be "days" or "months".
        duration_col: The accounting year duration column name.

    Returns:
        The count of declared accounting years for each given duration.

    """
    assert {"date_deb_exercice", "date_fin_exercice"} <= set(df.columns)

    ayd = accounting_year_duration(df, unit, duration_col)
    return ayd.groupBy(duration_col).count().orderBy(duration_col)


def accounting_year_duration(
    df: pyspark.sql.DataFrame, unit: str = "days", duration_col: str = "ay_duration"
) -> pyspark.sql.DataFrame:
    """Computes declared accounting year duration.

    Args:
        df: The input DataFrame. It must have a "date_deb_exercice" and
          "date_fin_exercice" columns.
        unit: The unit measuring accounting year duration. Should be "days" or "months".
        duration_col: The accounting year duration column name.

    Returns:
        A df with a new column of declared accounting years duration.

    """
    assert {"date_deb_exercice", "date_fin_exercice"} <= set(df.columns)

    if unit == "days":
        diff_function = F.datediff
    elif unit == "months":
        diff_function = F.months_between
    else:
        raise ValueError(f"Unknown unit {unit}")
    ayd = df.withColumn(
        duration_col,
        F.round(
            diff_function(
                F.to_date(df["date_fin_exercice"]),
                F.to_date(df["date_deb_exercice"]),
            )
        ).cast("int"),
    )
    return ayd


def is_centered(df: pyspark.sql.DataFrame, tol: float) -> Tuple[bool, List]:
    """Check if a DataFrame has a `features` column with centered individual variables.

    The `features` column is the result of at least a `VectorAssembler()`.

    Args:
        df : Input DataFrame.
        tol : A tolerance for the zero equality test.

    Returns:
        A couple consisting of:
          - A bool set to True if variables are centered else False.
          - A list of each column's mean.

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
