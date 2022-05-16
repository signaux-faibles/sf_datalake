"""Sampling functions."""

from typing import Tuple

import pyspark
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
        config: model configuration, as loaded by io.load_parameters().

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
        fraction=n_not_failing / (n_samples - n_failing), seed=config["SEED"]
    )
    oversampled_subset = failing_subset.union(not_failing_subset).persist(
        pyspark.StorageLevel.MEMORY_AND_DISK
    )

    # Split datasets according to dates and train/test split ratio.
    siren_train, siren_test = (
        df.select("siren")
        .distinct()
        .randomSplit(
            [config["TRAIN_TEST_SPLIT_RATIO"], 1 - config["TRAIN_TEST_SPLIT_RATIO"]],
            seed=config["SEED"],
        )
    )

    train = (
        oversampled_subset.filter(
            oversampled_subset["periode"] > config["TRAIN_DATES"][0]
        )
        .filter(oversampled_subset["periode"] < config["TRAIN_DATES"][1])
        .join(siren_train, how="inner", on="siren")
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    test = (
        df.filter(df["periode"] > config["TEST_DATES"][0])
        .filter(df["periode"] < config["TEST_DATES"][1])
        .join(siren_test, how="inner", on="siren")
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    prediction = df.filter(
        F.to_date(df["periode"]) == config["PREDICTION_DATE"]
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    return train, test, prediction
