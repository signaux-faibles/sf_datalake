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
