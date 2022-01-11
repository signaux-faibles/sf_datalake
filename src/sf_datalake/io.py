"""Utility functions for data handling."""

import logging
from os import path
from typing import Dict, Iterable, Optional

import pkg_resources
import pyspark.sql
from pyspark.sql import functions as F

import sf_datalake.utils


def load_data(
    data_paths: Dict[str, str], spl_ratio: float = None, seed: int = 1234
) -> Dict[str, pyspark.sql.DataFrame]:
    """Loads one or more orc-stored datasets and returns them in a dict.

    Args:
        data_paths: A dict[str, str] structured as follows: {dataframe_name: file_path}
          `dataframe_name` will be the key to use to get access to a given DataFrame in
          the returned dict.
        spl_ratio: If stated, the size of the return sampled datasets, as a fraction of
          the full datasets respective sizes.

    Returns:
        A dictionary of DataFrame objects.

    """
    datasets = {}

    spark = sf_datalake.utils.get_spark_session()
    for name, file_path in data_paths.items():
        df = spark.read.orc(file_path)
        if spl_ratio is not None:
            df = df.sample(fraction=spl_ratio, seed=seed)
        datasets[name] = df
    return datasets


def csv_to_orc(input_filename: str, output_filename: str):
    """Writes a file stored as csv in orc format.

    Args:
        input_filename: Path to a csv file.
        output_filename: Path to write the output orc file to.

    """
    spark = sf_datalake.utils.get_spark_session()
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


def write_predictions(
    output_dir: str,
    test_data: pyspark.sql.DataFrame,
    prediction_data: pyspark.sql.DataFrame,
):
    """Writes the results of a prediction to CSV files."""
    test_output_path = path.join(output_dir, "test_data.csv")
    prediction_output_path = path.join(output_dir, "prediction_data.csv")

    logging.info("Writing test data to file %s", test_output_path)
    test_data.select(
        ["siren", "time_til_failure", "failure_within_18m", "probability"]
    ).write.csv(test_output_path, header=True)

    logging.info("Writing prediction data to file %s", prediction_output_path)
    prediction_data.select(["siren", "probability"]).write.csv(
        prediction_output_path, header=True
    )


def write_explanations(
    output_dir: str,
    macro_scores_df: pyspark.sql.DataFrame,
    micro_scores_df: pyspark.sql.DataFrame,
):
    """Writes the explanations of a prediction to CSV files."""
    concerning_output_path = path.join(output_dir, "concerning_values.csv")
    explanation_output_path = path.join(output_dir, "explanation_data.csv")
    logging.info("Writing concerning features to file %s", concerning_output_path)
    micro_scores_df.write.csv(concerning_output_path, header=True)

    logging.info(
        "Writing explanation macro scores data to directory %s", explanation_output_path
    )
    macro_scores_df.write.csv(path.join(explanation_output_path), header=True)


def dump_configuration(
    output_dir: str, config: dict, dump_keys: Optional[Iterable] = None
):
    """Dumps a subset of the configuration used during a prediction run.

    Args:
        output_dir: The path where configuration should be dumped.
        config: Model configuration, as loaded by utils.get_config().
        dump_keys: An Iterable of configuration parameters that should be dumped.
          All elements of `dump_keys` must be part of `config`'s keys.

    """
    spark = sf_datalake.utils.get_spark_session()

    config["VERSION"] = pkg_resources.get_distribution("sf_datalake").version
    if dump_keys is None:
        dump_keys = (
            {
                "VERSION",
                "FILL_MISSING_VALUES",
                "TRAIN_TEST_SPLIT_RATIO",
                "TARGET_OVERSAMPLING_RATIO",
                "N_CONCERNING_MICRO",
                "TRAIN_DATES",
                "TEST_DATES",
                "PREDICTION_DATE",
                "MODEL",
                "FEATURES",
            },
        )
    sub_config = {k: v for k, v in config.items() if k in dump_keys}

    config_df = spark.createDataFrame(pyspark.sql.Row(sub_config))
    config_df.repartition(1).write.json(path.join(output_dir, "run_configuration.json"))
