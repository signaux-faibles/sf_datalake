import datetime as dt

import pytest
from pyspark.sql import types as T

from sf_datalake.transform import DateParser, IdentifierNormalizer, MissingValuesHandler


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


@pytest.fixture
def missing_value_handler_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ca", T.DoubleType(), True),
            T.StructField("ca_filled_value", T.DoubleType(), True),
            T.StructField("ca_filled_median", T.DoubleType(), True),
            T.StructField("ebe", T.DoubleType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("label", T.IntegerType(), True),
        ]
    )
    # fmt: off
    df = spark.createDataFrame(
        [('219385581', dt.date(2017, 12, 1), 35.0, 35.0, 35.0, 0.3034911450601422, '169', 0),
        ('219385581', dt.date(2022, 3, 1), 54.0, 54.0, 54.0, 0.08394427209402189, '347', 0),
        ('219385581', dt.date(2017, 3, 1), None, 0.0, 39.0,0.7482441546718624, '529', 0),
        ('219385581', dt.date(2015, 1, 1), None, 0.0, 39.0,0.8458750332248388, '006', 0),
        ('219385581', dt.date(2018, 1, 1), None, 0.0, 39.0, 0.05922352511577478, '631', 0),
        ('737745998', dt.date(2016, 6, 1), 6.0, 6.0, 6.0,0.23554547470210907, '366', 0),
        ('737745998', dt.date(2014, 10, 1), 39.0, 39.0, 39.0, 0.9144442485558925, '803', 0),
        ('737745998', dt.date(2015, 8, 1), 92.0, 92.0, 92.0,0.32033475571920367, '903', 1),
        ('737745998', dt.date(2015, 1, 1), None, 0.0, 39.0, 0.694039198363326, '944', 1),
        ('737745998', dt.date(2015, 1, 1), 76.0, 76.0, 76.0, 0.694039198363326, '944', 1),
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


def test_missing_value_handler(missing_value_handler_df):
    values = {"ca": 0}
    df = MissingValuesHandler(inputCols=["ca"], stat_strategy="median")._transform(
        missing_value_handler_df
    )
    df.show()
    assert all(r["ca"] == r["ca_filled_median"] for r in df.collect())
