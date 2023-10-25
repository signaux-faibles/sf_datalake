"""Model selection utilities."""

from typing import Tuple

import pyspark.sql


def train_test_split(
    df: pyspark.sql.DataFrame,
    random_seed: int,
    train_size: float = None,
    test_size: float = None,
    group_col: str = "siren",
) -> Tuple[
    pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame
]:  # pylint: disable=too-many-arguments, too-many-locals
    """Splits the input DataFrame and creates an oversampled training set.

    The data is split into 2 subsets: learn, test. The original set can be split
    in a stratified fashion and based on separate groups.

    Args:
        df: The DataFrame to split.
        random_seed: Controls the random sampling process.
        train_size: Should be between 0.0 and 1.0 and represent the proportion of the
          dataset to include in the train split. If None, will be set to 0.8.
        test_size: Should be between 0.0 and 1.0 and represent the proportion of the
          dataset to include in the test split.
        group_col: If not None, the two sets won't share any common value for this
          column, which is therefore considered as a group label.

    Returns:
        A tuple of two DataFrame associated with the training and testing stages.

    """
    # Parse and check sizes
    if train_size is None:
        if test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        else:
            train_size = 1 - test_size
    else:
        if test_size is not None:
            assert (
                train_size + test_size == 1
            ), "train_size and test_size should sum to 1."
        else:
            test_size = 1 - train_size

    # Split according to train/test split ratio and group column, if set.
    if group_col is not None:
        group_train, group_test = (
            df.select(group_col)
            .distinct()
            .randomSplit(
                weights=[train_size, test_size],
                seed=random_seed,
            )
        )
        df_train = df.join(group_train, how="left_semi", on=group_col)
        df_test = df.join(group_test, how="left_semi", on=group_col)
    else:
        df_train, df_test = df.randomSplit(weights=[train_size, test_size])

    return df_train, df_test
