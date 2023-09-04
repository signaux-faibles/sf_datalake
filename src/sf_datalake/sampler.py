"""Sampling functions."""

from typing import Tuple

import pyspark.sql


# TODO: Refactor this into train / test and over / sub sampling.
# See issue #70
def train_test_predict_split(
    df: pyspark.sql.DataFrame,
    class_column: str,
    target_oversampling_ratio: float,
    train_test_split_ratio: float,
    train_dates: Tuple[str],
    test_dates: Tuple[str],
    prediction_date: str,
    random_seed: float,
) -> Tuple[
    pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame
]:  # pylint: disable=too-many-arguments, too-many-locals
    """Splits the input DataFrame and creates an oversampled training set.

    The data is split into 3 parts: learn, test, prediction. The learning set is
    sampled so that the target positive class is represented according to the ratio
    defined under the `target_oversampling_ratio` parameter.

    Training set and test set relative sizes are defined through the
    `train_test_split_ratio` parameter.

    Args:
        df: the DataFrame to sample
        # TODO: COMPLETE THIS

    Returns:
        A tuple of three DataFrame, each associated with the following stages: learn,
          test, prediction.

    """
    positive_class_mask = df[class_column].astype("boolean")

    n_samples = df.count()
    n_failing = df.filter(positive_class_mask).count()
    subset_size = int(n_failing / target_oversampling_ratio)
    n_not_failing = int((1.0 - target_oversampling_ratio) * subset_size)

    failing_subset = df.filter(positive_class_mask)
    not_failing_subset = df.filter(~positive_class_mask).sample(
        fraction=n_not_failing / (n_samples - n_failing), seed=random_seed
    )
    oversampled_subset = failing_subset.union(not_failing_subset)

    # Split datasets according to dates and train/test split ratio.
    siren_train, siren_test = (
        df.select("siren")
        .distinct()
        .randomSplit(
            [train_test_split_ratio, 1 - train_test_split_ratio],
            seed=random_seed,
        )
    )

    train = (
        oversampled_subset.filter(oversampled_subset["periode"] >= train_dates[0])
        .filter(oversampled_subset["periode"] <= train_dates[1])
        .join(siren_train, how="inner", on="siren")
    )
    test = (
        df.filter(df["periode"] >= test_dates[0])
        .filter(df["periode"] <= test_dates[1])
        .join(siren_test, how="inner", on="siren")
    )
    prediction = df.filter(df["periode"] == prediction_date)
    return train, test, prediction
