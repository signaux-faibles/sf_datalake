"""Fixtures that will be used during the whole pytest session.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder.master("local[2]")
        .appName("sf_datalake-unit-tests")
        .getOrCreate()
    )
