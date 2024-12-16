"""Utility functions for data exploration using spark DataFrame objects.
"""

from datetime import datetime
from typing import List, Tuple

import numpy as np
import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T
import scipy.stats
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

import sf_datalake.utils


def accounting_time_span(
    df: pyspark.sql.DataFrame,
) -> Tuple[pyspark.sql.Row, pyspark.sql.Row]:
    """Computes global time spans of companies accouting years.

    Args:
        df: The input DataFrame. It must have a "date_début_exercice" and
          "date_fin_exercice" columns.

    Returns:
        A couple of DataFrame objects, each containing the min and max dates associated
          with the beginning and end of accounting years ("exercice comptable").

    """
    assert {"date_début_exercice", "date_fin_exercice"} <= set(df.columns)

    date_deb_exercice_span = df.select(
        F.min("date_début_exercice"),
        F.max("date_début_exercice"),
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
        df: The input DataFrame. It must have a "date_début_exercice" and
          "date_fin_exercice" columns.
        unit: The unit measuring accounting year duration. Should be "days" or "months".
        duration_col: The accounting year duration column name.

    Returns:
        The count of declared accounting years for each given duration.

    """
    assert {"date_début_exercice", "date_fin_exercice"} <= set(df.columns)

    ayd = accounting_year_duration(df, unit, duration_col)
    return ayd.groupBy(duration_col).count().orderBy(duration_col)


def accounting_year_duration(
    df: pyspark.sql.DataFrame, unit: str = "days", duration_col: str = "ay_duration"
) -> pyspark.sql.DataFrame:
    """Computes declared accounting year duration.

    Args:
        df: The input DataFrame. It must have a "date_début_exercice" and
          "date_fin_exercice" columns.
        unit: The unit measuring accounting year duration. Should be "days" or "months".
        duration_col: The accounting year duration column name.

    Returns:
        A df with a new column of declared accounting years duration.

    """
    assert {"date_début_exercice", "date_fin_exercice"} <= set(df.columns)

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
                F.to_date(df["date_début_exercice"]),
            )
        ).cast("int"),
    )
    return ayd


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

    udf_squared_diff = F.udf(lambda x: x[0] * (x[1] - x[2]) ** 2, T.DoubleType())
    df_squared_diff = df_groups.withColumn(
        "squared_diff",
        udf_squared_diff(F.struct("nobs_per_group", "global_avg", "group_avg")),
    )
    ssbg = df_squared_diff.select(F.sum("squared_diff")).take(1)[0][0]

    udf_within_ss = F.udf(lambda x: (x[0] - 1) * x[1] ** 2, T.DoubleType())
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


def build_eigenspace(df: pyspark.sql.DataFrame, features: List[str], k: int) -> dict:
    """Build the eigenspace of a DataFrame.

    Args:
        df: input DataFrame
        features: names of the features used to build the eigenspace
        k: dimension of the eigenspace

    Returns:
        Eigenvalues, inverse of eigenvalues, eigenvectors, variance explained/
    """
    assert set(features) <= set(df.columns)

    df = df.select(features)
    mat = RowMatrix(df.rdd.map(Vectors.dense))
    svd = mat.computeSVD(len(features))
    s_squared = [x * x for x in list(svd.s)]
    return {
        "s": svd.s[0:k],
        "explained_variance": np.cumsum(s_squared / sum(s_squared))[k - 1],
        "s_inverse": np.diag([1 / x for x in svd.s[0:k]]),
        "V": svd.V.toArray()[:, 0:k],
    }


def project_on_eigenspace(
    df: pyspark.sql.DataFrame, eigenspace: dict, features: List[str]
) -> List[List[float]]:
    """Project a DataFrame with a list of features on an eigenspace.

    Args:
        df: input DataFrame
        eigenspace: obtained from the function build_eigenspace()
        features : names of the features used to build the eigenspace

    Returns:
        The list of new coordinates corresponding to the DataFrame
        projected on the eigenspace.
    """
    assert set(features) <= set(df.columns)
    assert "s_inverse" in eigenspace.keys()
    assert "V" in eigenspace.keys()

    df = df.select(features)
    X = np.array(df.collect())
    U = np.matmul(X, np.matmul(eigenspace["V"], eigenspace["s_inverse"]))
    return [[float(y) for y in x] for x in U]


def convert_projections_to_dataframe(
    U: List[List[float]], df: pyspark.sql.DataFrame, period: datetime
) -> pyspark.sql.DataFrame:
    """Convert projections as a List[List[float]] into a DataFrame.
    This function is internally used by project_on_eigenspace_over_time().
    It is limited to eigenvectors of dimension 2.

    Args:
        U: eigenvectors
        df: the original DataFrame that will be fed with the eigenvectors
        period: period of the projection

    Returns:
        A DataFrame where features has been replaced by eigenvectors.
    """
    assert "siren" in set(df.columns)

    all_siren = df.select("siren").rdd.flatMap(lambda x: x).collect()
    U_on_cp1 = [x[0] for x in U]
    U_on_cp2 = [x[1] for x in U]

    data = [
        (siren, period, cp1, cp2)
        for siren, cp1, cp2 in zip(all_siren, U_on_cp1, U_on_cp2)
    ]
    spark = sf_datalake.utils.get_spark_session()
    rdd = spark.sparkContext.parallelize(data)
    return rdd.toDF(["siren", "période", "cp1", "cp2"])


def project_observations_on_eigenspace_over_time(
    df: pyspark.sql.DataFrame, start: str, end: str, features: List[str]
) -> pyspark.sql.DataFrame:
    """Build an eigenspace from the first period `start` and project the
    data from all next periods on this space.

    Args:
        df: input DataFrame
        start: start of the period as 'yyyy-mm-dd'
        end: end of the period as 'yyyy-mm-dd'
        features: names of the features used

    Returns:
        Data over time projected for each period on the eigenspace built
        from the first period.
    """
    assert set(["siren", "période"] + features) <= set(df.columns)

    df_pca = (
        df.filter(df["période"] >= start)
        .filter(df.période < end)
        .orderBy(["période", "siren"])
    )
    periods = df_pca.select("période").distinct().rdd.flatMap(lambda x: x).collect()
    eigenspace = build_eigenspace(
        df_pca.filter(df_pca.période == periods[0]), features, 2
    )

    spark = sf_datalake.utils.get_spark_session()

    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("période", T.TimestampType(), True),
            T.StructField("cp1", T.DoubleType(), True),
            T.StructField("cp2", T.DoubleType(), True),
        ]
    )
    obs_trajectories = spark.createDataFrame(
        data=spark.sparkContext.emptyRDD(), schema=schema
    )

    for period in periods:  # groupBy() to optimize?
        df_period = df_pca.filter(df.période == period)
        U_period = project_on_eigenspace(df_period, eigenspace, features)
        df_period_eigenspace = convert_projections_to_dataframe(
            U_period, df_period, period
        )
        obs_trajectories = obs_trajectories.union(df_period_eigenspace)

    return obs_trajectories


def convert_features_projection_to_dataframe(
    V: List[List[float]], features: List[str], period: datetime
) -> pyspark.sql.DataFrame:
    """Convert projections as a List[List[float]] into a DataFrame.
    This function is internally used by build_features_on_eigenspace_over_time().
    It is limited to eigenvectors of dimension 2.

    Args:
        U: eigenvectors
        features: Names of the features in the same order as in the argument
            features used in build_eigenspace()
        period: period of the projection

    Returns:
        A DataFrame where features are represented in an eigenspace.
    """
    V = [[float(y) for y in x] for x in V]
    V_on_cp1 = [x[0] for x in V]
    V_on_cp2 = [x[1] for x in V]

    data = [
        (feat, period, cp1, cp2) for feat, cp1, cp2 in zip(features, V_on_cp1, V_on_cp2)
    ]
    spark = sf_datalake.utils.get_spark_session()
    rdd = spark.sparkContext.parallelize(data)
    return rdd.toDF(["feature", "période", "cp1", "cp2"])


def project_features_on_eigenspace_over_time(
    df: pyspark.sql.DataFrame, start: str, end: str, features: List[str]
) -> pyspark.sql.DataFrame:
    """Build features projections on eigenspace for each period.

    Args:
        df: input DataFrame
        start: start of the period as 'yyyy-mm-dd'
        end: end of the period as 'yyyy-mm-dd'
        features: names of the features used

    Returns:
        Features over time projected on the eigenspace for each period.
    """
    assert set(["siren", "période"] + features) <= set(df.columns)

    df_pca = (
        df.filter(df.période >= start)
        .filter(df.période < end)
        .orderBy(["période", "siren"])
        .select(features + ["période", "siren"])
    )
    periods = df_pca.select("période").distinct().rdd.flatMap(lambda x: x).collect()

    spark = sf_datalake.utils.get_spark_session()

    schema = T.StructType(
        [
            T.StructField("feature", T.StringType(), True),
            T.StructField("période", T.TimestampType(), True),
            T.StructField("cp1", T.DoubleType(), True),
            T.StructField("cp2", T.DoubleType(), True),
        ]
    )
    features_trajectories = spark.createDataFrame(
        data=spark.sparkContext.emptyRDD(), schema=schema
    )

    for period in periods:  # groupBy() to optimize?
        eigenspace = build_eigenspace(
            df_pca.filter(df_pca.période == period), features, 2
        )
        V_period = eigenspace["V"]
        df_period_eigenspace = convert_features_projection_to_dataframe(
            V_period, features, period
        )
        features_trajectories = features_trajectories.union(df_period_eigenspace)

    return features_trajectories


def qqplot(
    df1: pyspark.sql.DataFrame,
    df2: pyspark.sql.DataFrame,
    feature: str,
    quantiles: List[str] = [f"{i}%" for i in range(5, 96)],
) -> pyspark.sql.DataFrame:
    """Generate the dataset ready to produce a Q-Q plot.

    This will produce Q-Q plot data about a given `feature` found in two different
    datasets.

    Args:
        df1: input DataFrame with `feature_name` as a column
        df2: input DataFrame with `feature_name` as a column
        feature_name: name of the feature in both df1 and df2 DataFrames
        quantiles: list of the quantiles to be computed

    Returns:
        A DataFrame with 3 columns:
            - `summary`: the quantiles
            - `x`: values of the quantiles for the feature `feature_name` from df1
            - `y`: values of the quantiles for the feature `feature_name` from df2
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
