import datetime as dt
import random

import pytest
from pyspark.sql import types as T

from sf_datalake.transform import DateParser, IdentifierNormalizer, RandomResampler


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
        [('219385581', dt.date(2017, 12, 1), 35, 0.3034911450601422, '169', 0),
        ('219385581', dt.date(2022, 3, 1), 54, 0.08394427209402189, '347', 0),
        ('219385581', dt.date(2017, 3, 1), 41, 0.7482441546718624, '529', 0),
        ('219385581', dt.date(2015, 1, 1), 48, 0.8458750332248388, '006', 0),
        ('219385581', dt.date(2018, 1, 1), 71, 0.05922352511577478, '631', 0),
        ('737745998', dt.date(2016, 6, 1), 6, 0.23554547470210907, '366', 0),
        ('737745998', dt.date(2014, 10, 1), 39, 0.9144442485558925, '803', 0),
        ('737745998', dt.date(2015, 8, 1), 92, 0.32033475571920367, '903', 0),
        ('737745998', dt.date(2015, 1, 1), 66, 0.694039198363326, '944', 0),
        ('737745998', dt.date(2019, 8, 1), 76, 0.1680343105355956, '808', 0),
        ('614889185', dt.date(2019, 2, 1), 79, 0.16736434937871758, '521', 0),
        ('614889185', dt.date(2015, 3, 1), 87, 0.8089011463158653, '250', 0),
        ('614889185', dt.date(2016, 1, 1), 48, 0.5864655437098659, '358', 0),
        ('614889185', dt.date(2016, 8, 1), 15, 0.6027506727512615, '883', 0),
        ('614889185', dt.date(2020, 8, 1), 99, 0.21462396555702568, '017', 0),
        ('186331439', dt.date(2019, 8, 1), 18, 0.05994703380143718, '711', 0),
        ('186331439', dt.date(2022, 2, 1), 86, 0.7503833614639871, '708', 0),
        ('186331439', dt.date(2019, 10, 1), 26, 0.7224095021086719, '615', 0),
        ('186331439', dt.date(2023, 6, 1), 15, 0.8051712533202542, '408', 1),
        ('186331439', dt.date(2017, 7, 1), 84, 0.3989912559412453, '195', 1),
        ('980933722', dt.date(2018, 8, 1), 9, 0.01798403220657474, '103', 1),
        ('980933722', dt.date(2023, 10, 1), 6, 0.6063299987951098, '995', 1),
        ('980933722', dt.date(2018, 7, 1), 91, 0.3801052159050209, '778', 0),
        ('980933722', dt.date(2018, 11, 1), 42, 0.8761944708456222, '331', 0),
        ('980933722', dt.date(2017, 3, 1), 21, 0.3973330408443837, '488', 0)
        ],
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
