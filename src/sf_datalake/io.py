"""Utility functions for data handling."""

import argparse
import json
import logging
from os import path
from typing import Dict, Iterable, Optional

import pkg_resources
import pyspark.sql

import sf_datalake.utils


def data_path_parser() -> argparse.ArgumentParser:
    """Creates a general argument parser for data file / directories io handling.

    Two positional arguments are added without default value:
    1) data input directory.
    2) data output directory.

    Returns:
        An ArgumentParser object ready to be used as is or further customized.

    """
    parser = argparse.ArgumentParser(description="General parser for data files io.")
    parser.add_argument(
        "input",
        help="""Path to an input source. Can be either:
        - A directory containing different files.
        - A single file.

        These should be readable by a pyspark.sql.DataFrameReader subclass instance.
        """,
    )
    parser.add_argument(
        "output",
        help="""Output path where the output dataset(s) will be stored.""",
    )
    return parser


def load_data(
    data_paths: Dict[str, str],
    file_format: str = None,
    sep: str = ",",
    spl_ratio: float = None,
    seed: int = 1234,
) -> Dict[str, pyspark.sql.DataFrame]:
    """Loads one or more orc-stored datasets and returns them in a dict.

    Args:
        data_paths: A dict[str, str] structured as follows: {dataframe_name: file_path}
          `dataframe_name` will be the key to use to get access to a given DataFrame in
          the returned dict.
        file_format: The file format, can be either "csv" or "orc".
        sep: Separator character, in case `file_format` is "csv".
        spl_ratio: If stated, the size of the return sampled datasets, as a fraction of
          the full datasets respective sizes.
        seed: A random seed, used for sub-sampling in case spl_ratio is < 1.

    Returns:
        A dictionary of datasets as pyspark DataFrame objects.

    """
    datasets = {}

    spark = sf_datalake.utils.get_spark_session()
    for name, file_path in data_paths.items():
        if file_format is None:
            file_format = path.splitext(file_path)[-1][1:]
        if file_format == "csv":
            df = spark.read.csv(file_path, sep=sep, inferSchema=True, header=True)
        elif file_format == "orc":
            df = spark.read.orc(file_path)
        else:
            raise ValueError(f"Unknown file format {file_format}.")
        if spl_ratio is not None:
            df = df.sample(fraction=spl_ratio, seed=seed)
        datasets[name] = df
    return datasets


def csv_to_orc(input_filename: str, output_filename: str, sep: str):
    """Writes a file stored as csv in orc format.

    Args:
        input_filename: Path to a csv file.
        output_filename: Path to write the output orc file to.
        sep: Separator character.

    """
    spark = sf_datalake.utils.get_spark_session()
    df = spark.read.csv(
        input_filename,
        sep=sep,
        inferSchema=True,
        header=True,
    )
    df.write.format("orc").save(output_filename)


def write_predictions(
    output_dir: str,
    test_data: pyspark.sql.DataFrame,
    prediction_data: pyspark.sql.DataFrame,
    n_rep: int = 5,
):
    """Writes the results of a prediction to CSV files."""
    test_output_path = path.join(output_dir, "test_data.csv")
    prediction_output_path = path.join(output_dir, "prediction_data.csv")

    logging.info("Writing test data to file %s", test_output_path)
    test_data.select(
        ["siren", "time_til_failure", "failure_within_18m", "probability"]
    ).repartition(n_rep).write.csv(test_output_path, header=True)

    logging.info("Writing prediction data to file %s", prediction_output_path)
    prediction_data.select(["siren", "probability"]).repartition(n_rep).write.csv(
        prediction_output_path, header=True
    )


def write_explanations(
    output_dir: str,
    macro_scores_df: pyspark.sql.DataFrame,
    micro_scores_df: pyspark.sql.DataFrame,
    n_rep: int = 5,
):
    """Writes the explanations of a prediction to CSV files."""
    concerning_output_path = path.join(output_dir, "concerning_values.csv")
    explanation_output_path = path.join(output_dir, "explanation_data.csv")
    logging.info("Writing concerning features to file %s", concerning_output_path)
    micro_scores_df.repartition(n_rep).write.csv(concerning_output_path, header=True)

    logging.info(
        "Writing explanation macro scores data to directory %s", explanation_output_path
    )
    macro_scores_df.repartition(n_rep).write.csv(
        path.join(explanation_output_path), header=True
    )


def load_parameters(fname: str) -> dict:
    """Loads a model run parameters from a preset config json file.

    Args:
        fname: Basename of a config file (including .json extension).

    Returns:
        The parameters to be used.

    """

    with pkg_resources.resource_stream(
        "sf_datalake", f"config/parameters/{fname}"
    ) as f:
        config = json.load(f)
    return config


def load_variables(fname: str) -> dict:
    """Loads a list of variables / features from a preset config json file.

    Args:
        fname: Basename of a config file (including .json extension).

    Returns:
        The variables, features and corresponding values .

    """

    with pkg_resources.resource_stream("sf_datalake", f"config/variables/{fname}") as f:
        config = json.load(f)
    return config


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
        dump_keys = {
            "SEED",
            "SAMPLE_RATIO",
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
        }
    sub_config = {k: v for k, v in config.items() if k in dump_keys}

    config_df = spark.createDataFrame(pyspark.sql.Row(sub_config))
    config_df.repartition(1).write.json(path.join(output_dir, "run_configuration.json"))
