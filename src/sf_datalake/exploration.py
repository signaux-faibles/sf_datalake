"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import Iterable, List, Tuple

import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import scipy.stats
from pyspark.sql.types import ArrayType, DoubleType, FloatType


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


def one_way_anova(
    df: pyspark.sql.DataFrame, categorical_var: str, continuous_var: str
) -> dict:
    """Compute the one-way ANOVA using the given categorical and continuous variables.

    Args:
        df : Input DataFrame.
        categorical_var : Name of the grouping feature. It has to be a column of
            boolean as integer.
        continuous_var : Name of the feature to analyze.

    Returns:
        F statistic, p-value, sum of squares within groups, sum of squares between
        groups, degrees of freedom within groups, degrees of freedom between groups.
    """
    df_groups = df.groupby(categorical_var).agg(
        F.avg(continuous_var).alias("group_avg"),
        F.stddev(continuous_var).alias("group_sse"),
        F.count("*").alias("nobs_per_group"),
    )

    global_avg = df.select(F.avg(continuous_var)).take(1)[0][0]
    df_groups = df_groups.withColumn("global_avg", F.lit(global_avg))

    udf_squared_diff = F.udf(lambda x: x[0] * (x[1] - x[2]) ** 2, DoubleType())
    df_squared_diff = df_groups.withColumn(
        "squared_diff",
        udf_squared_diff(F.struct("nobs_per_group", "global_avg", "group_avg")),
    )
    ssbg = df_squared_diff.select(F.sum("squared_diff")).take(1)[0][0]

    udf_within_ss = F.udf(lambda x: (x[0] - 1) * x[1] ** 2, DoubleType())
    df_squared_diff = df_groups.withColumn(
        "squared_within", udf_within_ss(F.struct("nobs_per_group", "group_sse"))
    )
    sswg = df_squared_diff.select(F.sum("squared_within")).take(1)[0][0]

    df_bg = df_groups.count() - 1
    df_wg = df.count() - df_groups.count() - 1

    f_statistic = (ssbg / df_bg) / (sswg / df_wg)
    p_value = 1 - scipy.stats.f.cdf(f_statistic, df_bg, df_wg)
    return {
        "f_statistic": f_statistic,
        "p_value": p_value,
        "sswg": sswg,
        "ssbg": ssbg,
        "df_wg": df_wg,
        "df_bg": df_bg,
    }
