"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import Iterable, Tuple

import pyspark  # pylint: disable=E0401
import pyspark.sql.functions as F  # pylint: disable=E0401


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
            F.count(F.when(F.isnull(c), c)).alias(c)
            for (c, c_type) in df.dtypes
            if c_type not in omit_type
        ]
    )


def acounting_time_span(
    df: pyspark.sql.DataFrame,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Computes global time spans of companies accouting years.

    Args:
        df: The input DataFrame. It must have a "date_deb_exercice" and
          "date_fin_exercice".

    Returns:
        A couple of DataFrame objects, each containing the min and max dates associated
          with the beginning and end of accounting years ("exercice comptable").

    """
    date_deb_exercice_span = df.select(
        min("date_deb_exercice"),
        max("date_deb_exercice"),
    ).first()
    date_fin_exercice_span = df.select(
        min("date_fin_exercice"),
        max("date_fin_exercice"),
    ).first()

    return date_deb_exercice_span, date_fin_exercice_span


def accounting_duration(
    df: pyspark.sql.DataFrame, unit: str = "days"
) -> pyspark.sql.DataFrame:
    """Computes number of days / months in the declared accounting year.

    Args:
        df: The input DataFrame. It must have a "date_deb_exercice" and
          "date_fin_exercice".
        unit: The unit measuring accounting year duration. Should be "days" or "months".

    Returns:
        A DataFrame grouping the number of accounting years associated with a given
          duration in days / months

    """
    if unit == "days":
        avg_duration = df.select(
            F.datediff(
                F.to_date(df["date_fin_exercice"]),
                F.to_date(df["date_deb_exercice"]),
            ).alias("exercice_datediff")
        )
    elif unit == "months":
        avg_duration = df.select(
            F.months_between(
                F.to_date(df["date_fin_exercice"]),
                F.to_date(df["date_deb_exercice"]),
            ).alias("exercice_datediff")
        )
    else:
        raise ValueError(f"Unknown unit {unit}")
    return (
        avg_duration.withColumn(
            "datediff_floored", F.round("exercice_datediff").cast("int")
        )
        .groupBy("datediff_floored")
        .count()
        .orderBy("datediff_floored")
    )


def oversample_df(
    df: pyspark.sql.DataFrame, label_colname: str
) -> pyspark.sql.DataFrame:
    """Implements dataset oversampling using duplication.

    Args:
        df: The input DataFrame.
        label_colname: The name of the column holding binary 0/1 target labels. It is
           assumed that samples with labels 1 are the ones that should be oversampled.

    Returns:
        The oversampled dataset.

    """
    major_df = df.filter(df[label_colname] == 0)
    minor_df = df.filter(df[label_colname] == 1)
    ratio = int(major_df.count() / minor_df.count())

    # Duplicate the minority rows
    oversampled_df = minor_df.withColumn(
        "dummy", F.explode(F.array([F.lit(x) for x in range(ratio)]))
    ).drop("dummy")

    # Combine both oversampled minority rows and previous majority rows
    combined_df = major_df.unionAll(oversampled_df)
    return combined_df
