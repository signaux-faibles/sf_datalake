"""Utility functions for data handling.

"""
from os import path
from typing import Dict

import pyspark.sql  # pylint: disable=E0401
from pyspark.sql import SparkSession  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401


def instantiate_spark_session():
    """Creates or gets a SparkSession object."""
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.shuffle.blockTransferService", "nio")
    spark.conf.set("spark.driver.maxResultSize", "1300M")
    return spark


def load_data(
    data_paths: Dict[str, str], spl_size: float = None
) -> Dict[str, pyspark.sql.DataFrame]:
    """Loads one or more orc-stored datasets and returns them in a dict.

    Args:
        data_paths: A dict[str, str] structured as follows: {dataframe_name: file_path}
          `dataframe_name` will be the key to use to get access to a given DataFrame in
          the returned dict.
        spl_size: If stated, the size of the return sampled datasets, as a fraction of
          the full datasets respective sizes.

    Returns:
        A dictionary of DataFrame objects.

    """
    datasets = {}

    spark = instantiate_spark_session()
    for name, file_path in data_paths.items():
        df = spark.read.orc(file_path)
        if spl_size is not None:
            df = df.sample(spl_size)
        datasets[name] = df
    return datasets


def csv_to_orc(input_filename: str, output_filename: str):
    """Writes a file stored as csv in orc format.

    Args:
        input_filename: Path to a csv file.
        output_filename: Path to write the output orc file to.

    """
    spark = instantiate_spark_session()
    df = spark.read.options(inferSchema="True", header="True", delimiter="|").csv(
        path.join(input_filename)
    )
    df.write.format("orc").save(output_filename)


def stringify_and_pad_siren(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Normalizes the input DataFrame "siren" entries.

    Args:
        df: A DataFrame with a "siren" column, whose type can be cast to string.

    Returns:
        A DataFrame with zeros-left-padded SIREN data, as string type.

    """
    assert "siren" in df.columns, "Input DataFrame doesn't have a 'siren' column."
    df = df.withColumn("siren", df["siren"].cast("string"))
    df = df.withColumn("siren", F.lpad(df["siren"], 9, "0"))
    return df
