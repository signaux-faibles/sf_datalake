import datetime as dt

import pytest
from pyspark.sql import types as T

from sf_datalake.transform import (
    DateParser,
    IdentifierNormalizer,
    LagOperator,
    RandomResampler,
)


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
def lag_operator_df(spark):
    schema = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ca", T.IntegerType(), True),
            T.StructField("ca_lag1m_wo", T.IntegerType(), True),
            T.StructField("ca_lag1m_bfill", T.IntegerType(), True),
            T.StructField("ca_lag1m_ffill", T.IntegerType(), True),
            T.StructField("ebe", T.DoubleType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("label", T.IntegerType(), True),
        ]
    )
    # fmt: off
    df = spark.createDataFrame(
        [
            ("043339338", dt.date(2018, 1, 1), 7, None, 7, None, 0.6354364559266611, "760", 0),
            ("043339338", dt.date(2018, 2, 1), 9, 7, 7, 7,0.46708377132745593, "971", 0),
            ("043339338", dt.date(2018, 3, 1), 83, 9, 9,9,0.5866119708529862, "880", 0),
            ("043339338", dt.date(2018, 4, 1), 76, 83, 83,83,0.9126640598227, "307", 1),
            ("043339338", dt.date(2018, 5, 1), 90, 76, 76,76,0.434687497902168, "121", 1),
            ("043339338", dt.date(2018, 6, 1), 64, 90, 90,90, 0.9526150841135487, "540", 0),
            ("043339338", dt.date(2018, 7, 1), 83, 64, 64,64, 0.9075422885370632, "527", 0),
            ("043339338", dt.date(2018, 8, 1), 87, 83, 83,83,0.9331836317697791, "806", 0),
            ("043339338", dt.date(2018, 9, 1), 68, 87, 87,87,0.8741663559131666, "979", 1),
            ("043339338", dt.date(2018, 10, 1), 21, 68, 68,68,0.6222276906194403, "387", 1),
            ("293736607", dt.date(2020, 1, 1), 97, None, 97,None, 0.3537846649936036, "107", 0),
            ("293736607", dt.date(2020, 2, 1), 96, 97, 97, 97,0.042232614177742156, "538", 1),
            ("293736607", dt.date(2020, 3, 1), 33, 96, 96, 96,0.434218505659813, "068", 1),
            ("293736607", dt.date(2020, 4, 1), None, 33, 33,33,0.17566080780501403, "315", 1),
            ("293736607", dt.date(2020, 5, 1), 99, None, 99,33,0.7003481341474471, "670", 0),
            ("293736607", dt.date(2020, 6, 1), 71, 99, 99,99, 0.6626475549979821, "246", 1),
            ("293736607", dt.date(2020, 7, 1), 19, 71, 71,71, 0.8235084769687906, "919", 0),
            ("293736607", dt.date(2020, 8, 1), 95, 19, 19, 19,0.3750939170060519, "806", 0),
            ("293736607", dt.date(2020, 9, 1), None, 95, 95, 95,0.9831245903009137, "070", 0),
            ("293736607", dt.date(2020, 10, 1), 38, None, None, 95, 0.6819102467208926, "782", 1),
        ],
        schema,
    )
    # fmt: on
    return df


@pytest.fixture
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
        ('186331439', dt.date(2023, 6, 1), 15, 0.8051712533202542, '408', 0),
        ('186331439', dt.date(2017, 7, 1), 84, 0.3989912559412453, '195', 0),
        ('980933722', dt.date(2018, 8, 1), 9, 0.01798403220657474, '103', 0),
        ('980933722', dt.date(2023, 10, 1), 6, 0.6063299987951098, '995', 0),
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


def test_lag_operator_wo(lag_operator_df):
    df_wo = LagOperator(inputCol="ca", n_months=1, bfill=False, ffill=False).transform(
        lag_operator_df
    )
    assert all(r["ca_lag1m_wo"] == r["ca_lag1m"] for r in df_wo.collect())


def test_lag_operator_bfill(lag_operator_df):
    df_bfill = LagOperator(
        inputCol="ca", n_months=1, bfill=True, ffill=False
    ).transform(lag_operator_df)
    assert all(r["ca_lag1m_bfill"] == r["ca_lag1m"] for r in df_bfill.collect())


def test_lag_operator_ffill(lag_operator_df):
    df_ffill = LagOperator(
        inputCol="ca", n_months=1, bfill=False, ffill=True
    ).transform(lag_operator_df)
    assert all(r["ca_lag1m_ffill"] == r["ca_lag1m"] for r in df_ffill.collect())
