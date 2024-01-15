"""Fixtures and utilities that will be used during the whole pytest session.
"""

import datetime as dt
import random
from typing import List, Tuple

import pyspark.sql
import pytest
from pyspark.sql import types as T

spark_session = (
    pyspark.sql.SparkSession.builder.master("local[2]")
    .appName("sf_datalake-unit-tests")
    .getOrCreate()
)


@pytest.fixture(scope="session")
def spark():
    return spark_session


class MockDataFrameGenerator:
    """Generates mock data for pyspark tests.

    A default schema is enforced, containing data of different types: integers, floats,
    dates (first day of month), strings. Two "index" columns are created: "siren" and
    "periode".

    Attributes:
        n_siren: the number of individual SIREN to mock.
        n_rows_per_siren: Number of different rows per SIREN.
        n_rows_perturbation: If not zero, will add or subtract a random number of rows
          for each subgroup of generated data associated with a SIREN.
        start_date: The lower bound for the "periode" column values.
        end_date: The upper bound for the "periode" column values.
        data: The generated dataframe

    Args:
        Same as Attributes, except `data`, that is generated at the end of instantiation.

    """

    def __init__(
        self,
        n_siren: int,
        n_rows_per_siren: int = 3,
        n_rows_perturbation: int = 0,
        start_date: str = "2014-01-01",
        end_date: str = "2024-01-01",
    ):
        self.n_siren = n_siren
        self.n_rows_per_siren = n_rows_per_siren
        self.n_rows_perturbation = n_rows_perturbation
        self.start_date = start_date
        self.end_date = end_date
        self.generate_mock_df()

    def mock_siren(self) -> str:
        return "".join(str(random.randint(0, 9)) for _ in range(9))

    def mock_date_ms(self) -> dt.date:
        """Generate a random date at month start.

        Returns:
            A date included in the [self.start_date, self.end_date] interval.

        """
        start_year = int(self.start_date.split("-")[0])
        start_month = int(self.start_date.split("-")[1])
        end_year = int(self.end_date.split("-")[0])
        end_month = int(self.end_date.split("-")[1])
        return dt.date(
            random.randint(start_year, end_year),
            random.randint(start_month, end_month),
            1,
        )

    def mock_integer(self) -> int:
        return random.randint(0, 100)

    def mock_float(self) -> float:
        return random.random()

    def mock_category(self) -> str:
        return "".join(str(random.randint(0, 9)) for _ in range(3))

    def mock_label(self) -> int:
        return random.randint(0, 1)

    def generate_mock_df(self):
        """Generate the mock DataFrame.

        It will set the `data` attribute. Each generated SIREN will have hold at least a
        row of data.
        """
        data: List[Tuple] = []
        for _ in range(self.n_siren):
            n_rows = self.n_rows_per_siren
            if self.n_rows_perturbation:
                n_rows += random.randint(
                    -self.n_rows_perturbation, self.n_rows_perturbation
                )
                n_rows = max(1, n_rows)
            siren = self.mock_siren()
            for _ in range(n_rows):
                data.append(
                    (
                        siren,
                        self.mock_date_ms(),
                        self.mock_integer(),
                        self.mock_float(),
                        self.mock_category(),
                        self.mock_label(),
                    )
                )
        schema = T.StructType(
            [
                T.StructField("siren", T.StringType(), nullable=False),
                T.StructField("periode", T.DateType(), nullable=False),
                T.StructField("ca", T.IntegerType(), nullable=True),
                T.StructField("ebe", T.DoubleType(), nullable=True),
                T.StructField("category", T.StringType(), nullable=True),
                T.StructField("label", T.IntegerType(), nullable=True),
            ]
        )
        self.data = spark_session.createDataFrame(data, schema)
