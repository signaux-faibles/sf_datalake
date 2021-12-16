"""Sampler class.

Sampler is an abstract class to build different samplers.
A sampler splits the input data in 3 parts: learn, test, prediction.
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pyspark.sql  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401

from sf_datalake.config import Config


class Sampler(ABC):  # pylint: disable=C0115, R0903
    def __init__(self, config: Config):
        self.config = config.get_config()

    @abstractmethod
    def run(
        self, df: pyspark.sql.DataFrame
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
        """Compute the sampling"""


class BaseSampler(Sampler):  # pylint: disable=R0903
    """Sampler by creating an oversampling training set"""

    def run(
        self, df: pyspark.sql.DataFrame
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:

        logging.info(
            "Creating oversampled training set with positive examples ratio %.1f",
            self.config["OVERSAMPLING_RATIO"],
        )

        will_fail_mask = df["failure_within_18m"]

        n_samples = df.count()
        n_failing = df.filter(will_fail_mask).count()
        subset_size = int(n_failing / self.config["OVERSAMPLING_RATIO"])
        n_not_failing = int((1.0 - self.config["OVERSAMPLING_RATIO"]) * subset_size)

        failing_subset = df.filter(will_fail_mask)
        not_failing_subset = df.filter(~will_fail_mask).sample(
            n_not_failing / (n_samples - n_failing)
        )
        oversampled_subset = failing_subset.union(not_failing_subset)

        # Define dates
        SIREN_train, SIREN_test = df.select("siren").distinct().randomSplit([0.8, 0.2])

        logging.info("Creating train between %s and %s.", *self.config["TRAIN_DATES"])
        train = (
            oversampled_subset.filter(
                oversampled_subset["siren"].isin(SIREN_train["siren"])
            )
            .filter(oversampled_subset["periode"] > self.config["TRAIN_DATES"][0])
            .filter(oversampled_subset["periode"] < self.config["TRAIN_DATES"][1])
        )

        logging.info("Creating test set between %s and %s.", *self.config["TEST_DATES"])
        test = (
            df.filter(df["siren"].isin(SIREN_test["siren"]))
            .filter(df["periode"] > self.config["TEST_DATES"][0])
            .filter(df["periode"] < self.config["TEST_DATES"][1])
        )

        logging.info("Creating a prediction set on %s.", self.config["PREDICTION_DATE"])
        prediction = df.filter(
            F.to_date(df["periode"]) == self.config["PREDICTION_DATE"]
        )
        return train, test, prediction


def factory_sampler(config: Config) -> Sampler:
    """Factory for samplers."""
    samplers = {"BaseSampler": BaseSampler}
    return samplers[config.get_config()["SAMPLER"]]
