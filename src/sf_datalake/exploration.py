"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import List, Tuple

import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T

import sf_datalake.transform
import sf_datalake.utils


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
) -> pyspark.sql.DataFrame:
    """Counts number of NaN values in numerical columns.

    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame specifying the number of NaN values in numerical fields.

    """
    return df.select(
        [
            F.count(F.when(F.isnull(c), c)).alias(c)
            for c in sf_datalake.utils.numerical_columns(df)
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
            "datediff_floored", F.round("exercice_datediff").cast(T.IntegerType())
        )
        .groupBy("datediff_floored")
        .count()
        .orderBy("datediff_floored")
    )


def is_centered(df: pyspark.sql.DataFrame, tol: float) -> Tuple[bool, List]:
    """Check if a DataFrame has a `features` column with centered individual variables.
    `features` column is the result of at least a `VectorAssembler()`.

    Args:
        df: Input DataFrame.
        tol: A tolerance for the zero equality test.

    Returns:
        Tuple[bool, List]: True if variables are centered else False. A list of the
                            mean of each variable.

    Example:
        is_centered(train_transformed.select(["features"]), tol = 1E-8)
    """
    assert "features" in df.columns, "Input DataFrame doesn't have a 'features' column."

    df = df.withColumn(
        "features_array", sf_datalake.utils.dense_to_array_udf("features")
    )
    n_features = len(df.first()["features"])

    df_agg = df.agg(
        F.array(*[F.avg(F.col("features_array")[i]) for i in range(n_features)]).alias(
            "mean"
        )
    )
    all_col_means = df_agg.select(F.col("mean")).collect()[0]["mean"]

    return (all(x < tol for x in all_col_means), all_col_means)


def qqplot_dataset(
    df1: pyspark.sql.DataFrame,
    df2: pyspark.sql.DataFrame,
    feature: str,
    quantiles: List[float] = [i / 100 for i in range(5, 96)],
) -> pyspark.sql.DataFrame:
    """Generate the dataset ready to produce a Q-Q plot.

    This will produce Q-Q plot data about a given `feature` found in two different
    datasets.

    Args:
        df1: Input DataFrame with feature as a column.
        df2: Input DataFrame with feature as a column.
        feature: Name of the feature in both df1 and df2.
        quantiles: List of the quantiles to be computed.

    Returns:
        A DataFrame with 3 columns:
          - `quantiles`: the quantiles
          - `x`: feature quantiles values in df1
          - `y`: feature quantiles values in df2

    """
    assert feature in df1.columns
    assert feature in df2.columns

    spark = sf_datalake.utils.get_spark_session()

    values = df1.approxQuantile(feature, quantiles, 0.001)
    dataset1 = spark.createDataFrame(list(zip(quantiles, values)), ["quantiles", "x"])

    values = df2.approxQuantile(feature, quantiles, 0.001)
    dataset2 = spark.createDataFrame(list(zip(quantiles, values)), ["quantiles", "y"])

    dataset = dataset1.join(dataset2, how="left", on="quantiles").orderBy("quantiles")
    return dataset


def print_spark_df_scores(results: pyspark.sql.DataFrame):
    """Quickly prints scores from data contained in a spark DataFrame."""
    correct_count = results.filter(results.label == results.prediction).count()
    total_count = results.count()
    correct_1_count = results.filter(
        (results.label == 1) & (results.prediction == 1)
    ).count()
    total_1_test = results.filter((results.label == 1)).count()
    total_1_predict = results.filter((results.prediction == 1)).count()

    print(f"All correct predections count: {correct_count}")
    print(f"Total count: {total_count}")
    print(f"Accuracy %: {(correct_count / total_count) * 100}")
    print(f"Recall %: {(correct_1_count / total_1_test) * 100}")
    print(f"Precision %: {(correct_1_count / total_1_predict) * 100}")
