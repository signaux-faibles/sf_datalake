import random

import pytest

from sf_datalake.model_selection import train_test_split
from tests.conftest import MockDataFrameGenerator


@pytest.fixture(scope="class")
def split_dataset():
    return MockDataFrameGenerator(
        n_siren=1000, n_rows_per_siren=3, n_rows_perturbation=1
    ).data


@pytest.mark.usefixtures("split_dataset")
class TestTrainTestSplit:
    def test_group_independance(self, split_dataset):
        train_data, test_data = train_test_split(
            df=split_dataset,
            random_seed=42,
            train_size=0.8,
            test_size=0.2,
            group_col="siren",
        )
        siren_intersection = train_data[["siren"]].intersect(test_data[["siren"]])
        assert (
            siren_intersection.count() == 0
        ), f"Intersection found: {siren_intersection.show()}"

    def test_test_subset_size(self, split_dataset):
        tolerance = 0.01
        train_size = 0.8
        test_size = 1 - train_size
        seed = random.randint(0, 1000)
        _, test_data = train_test_split(
            df=split_dataset,
            random_seed=seed,
            train_size=train_size,
            test_size=test_size,
            group_col="siren",
        )
        n_samples = split_dataset.count()
        assert (
            test_size - tolerance
            < test_data.count() / n_samples
            < test_size + tolerance
        )
