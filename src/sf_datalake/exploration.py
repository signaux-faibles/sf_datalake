"""Utility functions for data exploration using spark DataFrame objects.
"""

from typing import Iterable, List, Tuple

import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

import sf_datalake.transform


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

    dense_to_array_udf = F.udf(
        lambda v: [float(x) for x in v], T.ArrayType(T.FloatType())
    )

    df = df.withColumn("features_array", dense_to_array_udf("features"))
    n_features = len(df.first()["features"])

    df_agg = df.agg(
        F.array(*[F.avg(F.col("features_array")[i]) for i in range(n_features)]).alias(
            "mean"
        )
    )
    all_col_means = df_agg.select(F.col("mean")).collect()[0]["mean"]

    return (all(x < tol for x in all_col_means), all_col_means)


def generate_qqplot_dataset(
    df1: pyspark.sql.DataFrame,
    df2: pyspark.sql.DataFrame,
    feature: str,
    quantiles: List[str] = [f"{i}%" for i in range(5, 96)],
) -> pyspark.sql.DataFrame:
    """Generate the dataset ready to produce a Q-Q plot.

    Args:
        df1: input DataFrame with `feature` as a column
        df2: input DataFrame with `feature` as a column
        feature: name of the feature in both df1 and df2 DataFrames
        quantiles: list of the quantiles to be computed

    Returns:
        A DataFrame with 3 columns:
            - `summary`: the quantiles
            - `x`: values of the quantiles for the feature `feature` from df1
            - `y`: values of the quantiles for the feature `feature` from df2
    """
    assert feature in df1.columns
    assert feature in df2.columns

    df1 = (
        df1.summary(*quantiles)
        .select(["summary", feature])
        .withColumnRenamed(feature, "x")
    )
    df2 = (
        df2.summary(*quantiles)
        .select(["summary", feature])
        .withColumnRenamed(feature, "y")
    )
    df = df1.join(df2, how="left", on="summary")
    return df


def generate_unbiaser_covid_params(
    df: pyspark.sql.DataFrame, features: List[str], config: dict
) -> dict:
    """Generate for each feature in `features`, the necessary parameters to unbias
    the feature after the COVID-19 event.

    Args:
        df: input DataFrame
        features: name of the features to unbias
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A dict with features as keys. For each feature, the following dict:
            {
                "params": list of the parameters to unbias the feature
                "rmse": root mean square error of the unbias model for the feature
                "r2": r square of the unbias model for the feature
            }
    """
    pipeline_preprocessor = Pipeline(
        stages=sf_datalake.transform.generate_preprocessing_stages(config)
    )
    df = pipeline_preprocessor.fit(df).transform(df)

    # keep data from the first date of the learning dataset
    df = df.filter(df["periode"] >= config["TRAIN_DATES"][0])

    df1 = df.filter(df["periode"] <= "2020-02-29").select(features)
    df2 = df.filter(df["periode"] > "2020-02-29").select(features)

    unbiaser_params = {}
    for feat in features:
        df = generate_qqplot_dataset(df1, df2, feature=feat)
        df = df.withColumn("x", F.col("x").cast("float"))
        df = df.withColumn("y", F.col("y").cast("float"))
        vector_assembler = VectorAssembler(inputCols=["y"], outputCol="features")
        df_va = vector_assembler.transform(df).select(["features", "x"])

        lr = LinearRegression(featuresCol="features", labelCol="x")
        lr_model = lr.fit(df_va)
        unbiaser_params[feat] = {
            "params": [lr_model.intercept, lr_model.coefficients[0]],
            "rmse": lr_model.summary.rootMeanSquaredError,
            "r2": lr_model.summary.r2,
        }

    return unbiaser_params
