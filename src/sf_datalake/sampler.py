"""Sampling functions. """

from typing import Tuple

import pyspark.sql
from pyspark.sql import functions as F


def train_test_predict_split(
    df: pyspark.sql.DataFrame,
    config: dict,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Splits the input DataFrame and creates an oversampled training set.

    The data is split into 3 parts: learn, test, prediction. The learning set is
    sampled so that the target positive class is represented according to the ratio
    defined under the `TARGET_OVERSAMPLING_RATIO` config parameter.

    Training set and test set relative sizes are defined through the
    `TRAIN_TEST_SPLIT_RATIO` config parameter.

    Args:
        df: the DataFrame to sample
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A tuple of three DataFrame, each associated with the following stages: learn,
          test, prediction.

    """
    will_fail_mask = df["failure_within_18m"].astype("boolean")

    n_samples = df.count()
    n_failing = df.filter(will_fail_mask).count()
    subset_size = int(n_failing / config["TARGET_OVERSAMPLING_RATIO"])
    n_not_failing = int((1.0 - config["TARGET_OVERSAMPLING_RATIO"]) * subset_size)

    failing_subset = df.filter(will_fail_mask)
    not_failing_subset = df.filter(~will_fail_mask).sample(
        n_not_failing / (n_samples - n_failing)
    )
    oversampled_subset = failing_subset.union(not_failing_subset)

    # Split datasets according to dates and train/test split ratio.
    siren_train, siren_test = (
        df.select("siren")
        .distinct()
        .randomSplit(
            [config["TRAIN_TEST_SPLIT_RATIO"], 1 - config["TRAIN_TEST_SPLIT_RATIO"]]
        )
    )

    train = (
        oversampled_subset.filter(
            oversampled_subset["siren"].isin(siren_train["siren"])
        )
        .filter(oversampled_subset["periode"] > config["TRAIN_DATES"][0])
        .filter(oversampled_subset["periode"] < config["TRAIN_DATES"][1])
    )

    test = (
        df.filter(df["siren"].isin(siren_test["siren"]))
        .filter(df["periode"] > config["TEST_DATES"][0])
        .filter(df["periode"] < config["TEST_DATES"][1])
    )

    prediction = df.filter(F.to_date(df["periode"]) == config["PREDICTION_DATE"])
    return train, test, prediction
