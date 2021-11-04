"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import Iterable

import pyspark
import pyspark.sql.functions as F


def count_missing_values(df: pyspark.sql.DataFrame):
    """Counts number of null values in each column."""
    return df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])


def count_nan_values(
    df: pyspark.sql.DataFrame,
    omit_type: Iterable = ("timestamp", "string", "date", "bool"),
):
    """Counts number of nan values in numerical columns.

    Args:
        df : the input DataFrame.
        omit_type: an iterable containing the names of types that should not be looked
          for.

    Returns:
        A DataFrame listing the number of NaN values in numerical fields.
    """

    return df.select(
        [
            F.count(F.when(F.isnull(c), c)).alias(c)
            for (c, c_type) in df.dtypes
            if c_type not in omit_type
        ]
    )


def print_time_span(df: pyspark.sql.DataFrame):
    """Prints global time spans of companies accouting years."""
    date_deb_exercice_span = df.select(
        min("date_deb_exercice"),
        max("date_deb_exercice"),
    ).first()
    date_fin_exercice_span = df.select(
        min("date_fin_exercice"),
        max("date_fin_exercice"),
    ).first()

    print(
        f"Les dates de début d'exercice s'étendent de \
        {date_deb_exercice_span[0].strftime('%d/%m/%Y')} à \
        {date_deb_exercice_span[1].strftime('%d/%m/%Y')}"
    )
    print(
        f"Les dates de fin d'exercice s'étendent de \
        {date_fin_exercice_span[0].strftime('%d/%m/%Y')} à \
        {date_fin_exercice_span[1].strftime('%d/%m/%Y')}"
    )


def oversample_df(df: pyspark.sql.DataFrame, label_colname: str):
    """Implements dataset oversampling using duplication."""
    major_df = df.filter(df[label_colname] == 0)
    minor_df = df.filter(df[label_colname] == 1)
    ratio = int(major_df.count() / minor_df.count())
    a = range(ratio)

    # Duplicate the minority rows
    oversampled_df = minor_df.withColumn(
        "dummy", F.explode(F.array([F.lit(x) for x in a]))
    ).drop("dummy")

    # Combine both oversampled minority rows and previous majority rows
    combined_df = major_df.unionAll(oversampled_df)
    return combined_df
