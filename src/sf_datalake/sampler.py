"""Sampling functions.

"""
from typing import Tuple

import pyspark.sql
from pyspark.sql import functions as F


def sample_df(
    df: pyspark.sql.DataFrame,
    config: dict,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Sample the input DataFrame by creating an oversampling training set. It
    splits the data in 3 parts: learn, test, prediction.

    Args:
        df: the DataFrame to sample
        config: the config parameters (see utils.get_config())

    Returns:
        3 DataFrames, one for each of the following part: learn, test, prediction.
    """
    will_fail_mask = df["failure_within_18m"].astype("boolean")

    n_samples = df.count()
    n_failing = df.filter(will_fail_mask).count()
    subset_size = int(n_failing / config["OVERSAMPLING_RATIO"])
    n_not_failing = int((1.0 - config["OVERSAMPLING_RATIO"]) * subset_size)

    failing_subset = df.filter(will_fail_mask)
    not_failing_subset = df.filter(~will_fail_mask).sample(
        n_not_failing / (n_samples - n_failing)
    )
    oversampled_subset = failing_subset.union(not_failing_subset)

    # Define dates
    SIREN_train, SIREN_test = df.select("siren").distinct().randomSplit([0.8, 0.2])

    train = (
        oversampled_subset.filter(
            oversampled_subset["siren"].isin(SIREN_train["siren"])
        )
        .filter(oversampled_subset["periode"] > config["TRAIN_DATES"][0])
        .filter(oversampled_subset["periode"] < config["TRAIN_DATES"][1])
    )

    test = (
        df.filter(df["siren"].isin(SIREN_test["siren"]))
        .filter(df["periode"] > config["TEST_DATES"][0])
        .filter(df["periode"] < config["TEST_DATES"][1])
    )

    prediction = df.filter(F.to_date(df["periode"]) == config["PREDICTION_DATE"])
    return train, test, prediction
