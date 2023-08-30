# pylint: disable=missing-function-docstring
"""Fixtures that will be used during the whole pytest session.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Fixture for creating a spark context."""
    return (
        SparkSession.builder.master("local[2]")
        .appName("sf_datalake-unit-tests")
        .getOrCreate()
    )
