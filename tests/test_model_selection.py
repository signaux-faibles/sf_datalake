import datetime as dt

import pytest
from pyspark.sql import types as T

from sf_datalake.model_selection import train_test_split


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
        [
            ("361277009", dt.date(2017, 11, 1), 81, 0.462595507496679, "755", 1),
            ("361277009", dt.date(2016, 3, 1), 84, 0.5710125631116676, "304", 1),
            ("779681095", dt.date(2014, 5, 1), 99, 0.6288750846033718, "764", 0),
            ("779681095", dt.date(2022, 9, 1), 51, 0.22018893875685264, "144", 1),
            ("166591555", dt.date(2021, 10, 1), 38, 0.4519904690474965, "369", 0),
            ("166591555", dt.date(2014, 10, 1), 82, 0.246999907052183, "613", 1),
            ("373921064", dt.date(2020, 7, 1), 10, 0.3331701759218788, "273", 1),
            ("373921064", dt.date(2020, 4, 1), 13, 0.044001110200020954, "873", 1),
            ("373921064", dt.date(2018, 1, 1), 66, 0.9032250349069598, "036", 0),
            ("254020538", dt.date(2017, 8, 1), 8, 0.6128784436233967, "347", 0),
            ("254020538", dt.date(2021, 9, 1), 49, 0.23821780577087037, "301", 0),
        ],
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
        tolerance = 0.03
        train_data, test_data = train_test_split(
            df=split_dataset,
            random_seed=42,
            train_size=0.8,
            test_size=0.2,
            group_col="siren",
        )
        n_samples = split_dataset.count()
        assert (test_data.count() / n_samples > 0.2 - tolerance) & (
            test_data.count() / n_samples < 0.2 + tolerance
        )
