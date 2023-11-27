import datetime as dt
import random

import pytest
from pyspark.sql import types as T

from sf_datalake.model_selection import train_test_split


# Function to generate example
def generate_siren():
    return "".join(str(random.randint(0, 9)) for _ in range(9))


def generate_period():
    l = [30 * i for i in range(2)]
    n_days = l[random.randint(0, len(l) - 1)]
    return dt.date(random.randint(2014, 2023), random.randint(1, 12), 1)


def generate_value_integer():
    return random.randint(0, 100)


def generate_category():
    return "".join(str(random.randint(0, 9)) for _ in range(3))


def generate_value_double():
    return random.random()


def generate_label():
    return random.randint(0, 1)


def generate_data(n_siren, n_lines_by_siren, add_noise=False):
    res = []
    for _ in range(n_siren):
        if add_noise:
            n_lines_by_siren += random.randint(-1, 1)
        s = generate_siren()
        for _ in range(n_lines_by_siren):
            res.append(
                (
                    s,
                    generate_period(),
                    generate_value_integer(),
                    generate_value_double(),
                    generate_category(),
                    generate_label(),
                )
            )
    return res


@pytest.fixture(scope="class")
def split_dataset(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ca", T.IntegerType(), True),
            T.StructField("ebe", T.DoubleType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("label", T.IntegerType(), True),
        ]
    )

    df = spark.createDataFrame(
        generate_data(n_siren=1000, n_lines_by_siren=3, add_noise=True),
        schema,
    )
    return df


@pytest.mark.usefixtures("split_dataset")
class TestTrainTestSplit:
    def test_split_dataset_group_independance(self, split_dataset):
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

    def test_split_dataset_train_test_size(self, split_dataset):
        tolerance = 0.01
        train_size = 0.8
        test_size = 1 - train_size
        seed = random.randint(0, 1000)
        train_data, test_data = train_test_split(
            df=split_dataset,
            random_seed=seed,
            train_size=train_size,
            test_size=test_size,
            group_col="siren",
        )
        n_samples = split_dataset.count()
        assert (test_data.count() / n_samples > test_size - tolerance) & (
            test_data.count() / n_samples < test_size + tolerance
        )
