import datetime as dt
import random

import pytest
from pyspark.sql import types as T

from sf_datalake.transform import (
    DateParser,
    IdentifierNormalizer,
    LagOperator,
    MissingValuesHandler,
    RandomResampler,
)
from tests.conftest import MockDataFrameGenerator


@pytest.fixture
def siren_padding_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.IntegerType(), False),
            T.StructField("padded_siren", T.StringType(), False),
        ]
    )
    df = spark.createDataFrame(
        [(524893758, "524893758"), (45378, "000045378"), (54489542, "054489542")],
        schema=schema,
    )
    return df


@pytest.fixture
def parsed_date_df(spark):
    schema = T.StructType(
        [
            T.StructField("raw_date", T.StringType(), False),
            T.StructField("ref_date", T.DateType(), False),
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
def random_resampler_df():
    return MockDataFrameGenerator(
        n_siren=10000, n_rows_per_siren=3, n_rows_perturbation=1
    ).data


@pytest.fixture(scope="class")
def lag_operator_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), False),
            T.StructField("période", T.DateType(), False),
            T.StructField("ca", T.IntegerType(), True),
            T.StructField("expected_ca_lag1m", T.IntegerType(), True),
        ]
    )
    # fmt: off
    df = spark.createDataFrame(
        [
            ("043339338", dt.date(2018, 1, 1),  7, None),
            ("043339338", dt.date(2018, 2, 1),  9, 7),
            ("043339338", dt.date(2018, 3, 1),  83, 9),
            ("043339338", dt.date(2018, 4, 1),  76, 83),
            ("043339338", dt.date(2018, 5, 1),  90, 76),
            ("043339338", dt.date(2018, 6, 1),  64, 90),
            ("043339338", dt.date(2018, 7, 1),  83, 64),
            ("043339338", dt.date(2018, 8, 1),  87, 83),
            ("043339338", dt.date(2018, 9, 1),  68, 87),
            ("043339338", dt.date(2018, 10, 1), 21, 68),
            ("293736607", dt.date(2020, 1, 1),  97, None),
            ("293736607", dt.date(2020, 2, 1),  96, 97),
            ("293736607", dt.date(2020, 3, 1),  33, 96),
            ("293736607", dt.date(2020, 4, 1),  None, 33),
            ("293736607", dt.date(2020, 5, 1),  99, None),
            ("293736607", dt.date(2020, 6, 1),  71, 99),
            ("293736607", dt.date(2020, 7, 1),  19, 71),
            ("293736607", dt.date(2020, 8, 1),  95, 19),
            ("293736607", dt.date(2020, 9, 1),  None, 95),
            ("293736607", dt.date(2020, 10, 1), 38, None)
        ],
        schema=schema,
    )
    # fmt: on
    return df


@pytest.fixture(scope="class", name="missing_value_handler_df")
def fixture_missing_value_handler_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), False),
            T.StructField("période", T.DateType(), False),
            T.StructField("ca", T.DoubleType(), True),
            T.StructField("ca_filled_value", T.DoubleType(), True),
            T.StructField("ca_filled_median", T.DoubleType(), True),
            T.StructField("ca_filled_bfill", T.DoubleType(), True),
            T.StructField("ca_filled_ffill", T.DoubleType(), True),
        ]
    )

    # fmt: off
    df = spark.createDataFrame(
        [
            ('219385581', dt.date(2015, 1,  1), None,  0.0, 39.0, 35.0, None),
            ('219385581', dt.date(2017, 3,  1), None,  0.0, 39.0, 35.0, None),
            ('219385581', dt.date(2017, 12, 1), 35.0, 35.0, 35.0, 35.0, 35.0),
            ('219385581', dt.date(2018, 1,  1), None,  0.0, 39.0, 54.0, 35.0),
            ('219385581', dt.date(2022, 3,  1), 54.0, 54.0, 54.0, 54.0, 54.0),
            ('737745998', dt.date(2014, 10, 1), 39.0, 39.0, 39.0, 39.0, 39.0),
            ('737745998', dt.date(2015, 1,  1), None,  0.0, 39.0, 76.0, 39.0),
            ('737745998', dt.date(2015, 2,  1), 76.0, 76.0, 76.0, 76.0, 76.0),
            ('737745998', dt.date(2015, 8,  1), 92.0, 92.0, 92.0, 92.0, 92.0),
            ('737745998', dt.date(2016, 6,  1),  6.0,  6.0,  6.0,  6.0,  6.0),
        ],
        schema=schema,
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
        tolerance = 0.2
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
        tolerance = 0.2
        min_class_ratio = 0.5
        seed = random.randint(1, 1000)
        undersampled_df = RandomResampler(
            class_col="label",
            seed=seed,
            min_class_ratio=min_class_ratio,
            method="undersampling",
        ).transform(random_resampler_df)
        self.check_balance(undersampled_df, min_class_ratio, tolerance)


@pytest.mark.usefixtures("missing_value_handler_df")
class TestMissingValueHandler:
    def test_filling_with_median(self, missing_value_handler_df):
        df = MissingValuesHandler(inputCols=["ca"], strategy="median").transform(
            missing_value_handler_df
        )
        assert all(r["ca"] == r["ca_filled_median"] for r in df.collect())

    def test_filling_with_bfill(self, missing_value_handler_df):
        df = MissingValuesHandler(inputCols=["ca"], strategy="bfill").transform(
            missing_value_handler_df
        )
        assert all(r["ca"] == r["ca_filled_bfill"] for r in df.collect())

    def test_filling_with_ffill(self, missing_value_handler_df):
        df = MissingValuesHandler(inputCols=["ca"], strategy="ffill").transform(
            missing_value_handler_df
        )
        assert all(r["ca"] == r["ca_filled_ffill"] for r in df.collect())

    def test_filling_with_value(self, missing_value_handler_df):
        value = {"ca": 0.0}
        df = MissingValuesHandler(inputCols=["ca"], value=value).transform(
            missing_value_handler_df
        )
        assert all(r["ca"] == r["ca_filled_value"] for r in df.collect())


@pytest.mark.usefixtures("lag_operator_df")
class TestLagOperator:
    def test_1m_lag(self, lag_operator_df):
        df_1m = LagOperator(
            inputCol="ca",
            n_months=1,
        ).transform(lag_operator_df)
        assert all(r["expected_ca_lag1m"] == r["ca_lag1m"] for r in df_1m.collect())
