import datetime as dt
import random

import numpy as np
import pytest
from pyspark.sql import types as T

from sf_datalake.transform import DateParser, IdentifierNormalizer, RandomResampler


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


def generate_label(proba=0.5):
    return int(np.random.choice([0, 1], p=[1 - proba, proba]))


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
                    generate_label(proba=0.05),
                )
            )
    return res


@pytest.fixture
def siren_padding_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.IntegerType(), True),
            T.StructField("padded_siren", T.StringType(), True),
        ]
    )
    df = spark.createDataFrame(
        [(524893758, "524893758"), (45378, "000045378"), (54489542, "054489542")],
        schema,
    )
    return df


@pytest.fixture
def parsed_date_df(spark):
    schema = T.StructType(
        [
            T.StructField("raw_date", T.StringType(), True),
            T.StructField("ref_date", T.DateType(), True),
        ]
    )

    df = spark.createDataFrame(
        [
            ("20171130", dt.date(*(int(s) for s in "2017-11-30".split("-")))),
            ("20171229", dt.date(*(int(s) for s in "2017-12-29".split("-")))),
            ("20171229", dt.date(*(int(s) for s in "2017-12-29".split("-")))),
            ("20171229", dt.date(*(int(s) for s in "2017-12-29".split("-")))),
            ("20171031", dt.date(*(int(s) for s in "2017-10-31".split("-")))),
        ],
        schema,
    )
    return df


@pytest.fixture(scope="class")
def random_resampler_df(spark):
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
    # fmt: off
    df = spark.createDataFrame(
        generate_data(n_siren=10000,n_lines_by_siren=3, add_noise=True)
        ,
        schema,
    )
    # fmt: on
    return df


def test_siren_padding(siren_padding_df):
    df = IdentifierNormalizer(inputCol="siren", n_pad=9).transform(siren_padding_df)
    assert all(r["siren"] == r["padded_siren"] for r in df.collect())


def test_date_parser(parsed_date_df):
    df = DateParser(
        inputCol="raw_date", outputCol="parsed_date", format="yyyyMMdd"
    ).transform(parsed_date_df)
    assert all(r["ref_date"] == r["parsed_date"] for r in df.collect())


@pytest.mark.usefixtures("random_resampler_df")
class TestRandomResampler:
    def check_balance(self, df, req_min_cls_ratio, tol):
        class_counts = df.groupBy("label").count().rdd.collectAsMap()
        minority_class_count = class_counts[1]
        majority_class_count = class_counts[0]
        counts = minority_class_count + majority_class_count
        assert (
            req_min_cls_ratio - tol
            < (minority_class_count / counts)
            < req_min_cls_ratio + tol
        )

    def test_class_balance_oversampling(self, random_resampler_df):
        tolerance = 0.1
        min_class_ratio = 0.4
        seed = random.randint(1, 1000)
        oversampled_df = RandomResampler(
            class_col="label",
            seed=seed,
            min_class_ratio=min_class_ratio,
            method="oversampling",
        ).transform(random_resampler_df)
        return self.check_balance(oversampled_df, min_class_ratio, tolerance)

    def test_class_balance_undersampling(self, random_resampler_df):
        tolerance = 0.1
        min_class_ratio = 0.5
        seed = random.randint(1, 1000)
        undersampled_df = RandomResampler(
            class_col="label",
            seed=seed,
            min_class_ratio=min_class_ratio,
            method="undersampling",
        ).transform(random_resampler_df)
        self.check_balance(undersampled_df, min_class_ratio, tolerance)
