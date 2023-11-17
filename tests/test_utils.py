import datetime as dt

import pytest
from pyspark.sql import types as T

from sf_datalake.utils import merge_asof


@pytest.fixture
def merge_asof_df(spark):
    schema1 = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ca", T.IntegerType(), True),
            T.StructField("category", T.StringType(), True),
        ]
    )
    # fmt: off
    df1 = spark.createDataFrame(
        [
            ("043339338", dt.date(2018, 1, 1), 7, "760"),
            ("043339338", dt.date(2018, 2, 1), 9, "971"),
            ("043339338", dt.date(2018, 3, 1), 83,  "880"),
            ("043339338", dt.date(2018, 4, 1), 76,"307"),
            ("043339338", dt.date(2018, 5, 1), 90,  "121"),
            ("043339338", dt.date(2018, 6, 1), 64,  "540"),
            ("043339338", dt.date(2018, 7, 1), 83,  "527"),
            ("043339338", dt.date(2018, 8, 1), 87,  "806"),
            ("043339338", dt.date(2018, 9, 1), 68,  "979"),
            ("043339338", dt.date(2018, 10, 1), 21,  "387"),
            ("293736607", dt.date(2020, 1, 1), 97,  "107"),
            ("293736607", dt.date(2020, 2, 1), 96,  "538"),
            ("293736607", dt.date(2020, 3, 1), 33,  "068"),
            ("293736607", dt.date(2020, 4, 1), None,  "315"),
            ("293736607", dt.date(2020, 5, 1), 99,  "670"),
            ("293736607", dt.date(2020, 6, 1), 71,  "246"),
            ("293736607", dt.date(2020, 7, 1), 19, "919"),
            ("293736607", dt.date(2020, 8, 1), 95,  "806"),
            ("293736607", dt.date(2020, 9, 1), None,  "070"),
            ("293736607", dt.date(2020, 10, 1), 38, "782"),
        ],
        schema1,
    )

    schema2 = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ebe", T.IntegerType(), True),

        ]
    )
    # fmt: off
    df2 = spark.createDataFrame(
        [
            ("043339338", dt.date(2018, 1, 1), 200),
            ("043339338", dt.date(2018, 6, 1), 40),
            ("293736607", dt.date(2019, 1, 1), 50),
            ("293736607", dt.date(2020, 2, 1), 70),
            ("293736607", dt.date(2020, 8, 1), 30),
        ],
        schema2,
    )
    schema3 = T.StructType(
        [
            T.StructField("siren", T.StringType(), True),
            T.StructField("periode", T.DateType(), True),
            T.StructField("ca", T.IntegerType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("ebe_backward", T.IntegerType(), True),
            T.StructField("ebe_forward", T.IntegerType(), True),
            T.StructField("ebe_nearest", T.IntegerType(), True)
        ]
    )
    # fmt: off
    df3 = spark.createDataFrame(
        [
            ("043339338", dt.date(2018, 1, 1), 7, "760", 200, 200, 200),
            ("043339338", dt.date(2018, 2, 1), 9, "971", 200, 40,200),
            ("043339338", dt.date(2018, 3, 1), 83,  "880", 200, 40,200),
            ("043339338", dt.date(2018, 4, 1), 76,"307", 200, 40,40),
            ("043339338", dt.date(2018, 5, 1), 90,  "121", 200, 40,40),
            ("043339338", dt.date(2018, 6, 1), 64,  "540", 40, 40, 40 ),
            ("043339338", dt.date(2018, 7, 1), 83,  "527", 40, None, 40),
            ("043339338", dt.date(2018, 8, 1), 87,  "806", 40, None, 40),
            ("043339338", dt.date(2018, 9, 1), 68,  "979", 40, None, 40),
            ("043339338", dt.date(2018, 10, 1), 21,  "387", 40, None, 40),
            ("293736607", dt.date(2020, 1, 1), 97,  "107", 50, 70, 70),
            ("293736607", dt.date(2020, 2, 1), 96,  "538", 70, 70, 70),
            ("293736607", dt.date(2020, 3, 1), 33,  "068", 70, 30, 70),
            ("293736607", dt.date(2020, 4, 1), None,  "315", 70, 30,70),
            ("293736607", dt.date(2020, 5, 1), 99,  "670", 70, 30,70),
            ("293736607", dt.date(2020, 6, 1), 71,  "246", 70, 30,30),
            ("293736607", dt.date(2020, 7, 1), 19, "919", 70, 30,30),
            ("293736607", dt.date(2020, 8, 1), 95,  "806", 30, 30,30),
            ("293736607", dt.date(2020, 9, 1), None,  "070", 30, None,30),
            ("293736607", dt.date(2020, 10, 1), 38, "782", 30, None,30),
        ],
        schema3,
    )
    # fmt: on
    return df1, df2, df3


def test_merge_asof_backward(merge_asof_df):
    df1, df2, df_merge = merge_asof_df
    df = merge_asof(
        df1, df2, on="periode", by="siren", tolerance=365, direction="backward"
    )
    df.show()
    assert all(
        r["ebe"] == r_merge["ebe_backward"]
        for r, r_merge in zip(df.collect(), df_merge.collect())
    )


def test_merge_asof_forward(merge_asof_df):
    df1, df2, df_merge = merge_asof_df
    df = merge_asof(
        df1, df2, on="periode", by="siren", tolerance=365, direction="forward"
    )
    df.show()
    assert all(
        r["ebe"] == r_merge["ebe_forward"]
        for r, r_merge in zip(df.collect(), df_merge.collect())
    )


def test_merge_asof_nearest(merge_asof_df):
    df1, df2, df_merge = merge_asof_df
    df = merge_asof(
        df1, df2, on="periode", by="siren", tolerance=365, direction="nearest"
    )
    df.show()
    assert all(
        r["ebe"] == r_merge["ebe_nearest"]
        for r, r_merge in zip(df.collect(), df_merge.collect())
    )
