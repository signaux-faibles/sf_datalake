# pylint: disable=missing-function-docstring, redefined-outer-name
"""Test the src/transform.py module.
"""

import pytest

from sf_datalake.transform import IdentifierNormalizer  # DateParser, SiretToSiren


@pytest.fixture
def siren_padding_df(spark):
    return spark.createDataFrame(
        [(524893758, "524893758"), (45378, "000045378"), (54489542, "054489542")],
        "siren: int, padded_siren: string",
    )


@pytest.fixture
def parsed_date_df(spark):
    return spark.createDataFrame(
        ["20171130", "2017-11-30"],
        ["20171229", "2017-12-29"],
        ["20171229", "2017-12-29"],
        ["20171229", "2017-12-29"],
        ["20171031", "2017-10-31"],
        "raw_date: string, parsed_date: datetime.date",
    )


def test_siren_padding(siren_padding_df):
    df = IdentifierNormalizer(inputCol="siren", n_pad=9).transform(siren_padding_df)
    assert all(r["siren"] == r["padded_siren"] for r in df.collect())
