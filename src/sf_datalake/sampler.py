"""Sampling functions."""

from typing import Tuple

import pyspark
import pyspark.sql


def train_test_split(
    df: pyspark.sql.DataFrame,
    config: dict,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Splits the input DataFrame.

    The data is split into 2 parts: learn, test. T

    Training set and test set relative sizes are defined through the
    `TRAIN_TEST_SPLIT_RATIO` config parameter.

    Args:
        df: the DataFrame to sample
        config: model configuration, as loaded by io.load_parameters().

    Returns:
        A tuple of three DataFrame, each associated with the following stages: learn,
          test, prediction.

    """
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
        df.filter(df["periode"] >= config["TRAIN_DATES"][0])
        .filter(df["periode"] <= config["TRAIN_DATES"][1])
        .join(siren_train, how="inner", on="siren")
    )
    test = (
        df.filter(df["periode"] >= config["TEST_DATES"][0])
        .filter(df["periode"] <= config["TEST_DATES"][1])
        .join(siren_test, how="inner", on="siren")
    )
    return train, test
