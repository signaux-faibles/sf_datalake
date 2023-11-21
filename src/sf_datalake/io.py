"""Utility functions for data handling."""

import argparse
import logging
from os import path
from typing import Dict

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


def write_data(
    dataset: pyspark.sql.DataFrame,
    output_path: str,
    file_format: str,
    sep: str = ",",
):
    """Loads one or more orc-stored datasets and returns them in a dict.

    Args:
        dataset: A dataset.
        output_path: The output path.
        file_format: The file format, can be either "csv" or "orc".
        sep: Separator character, in case `file_format` is "csv".

    """
    write_options = {"header": True, "sep": sep} if file_format == "csv" else {}
    dataset.write.format(file_format).options(**write_options).save(output_path)


def load_data(
    data_paths: Dict[str, str],
    file_format: str = None,
    sep: str = ",",
    infer_schema: bool = True,
) -> Dict[str, pyspark.sql.DataFrame]:
    """Loads one or more datasets and returns them through a dict.

    Args:
        data_paths: A dict[str, str] structured as follows: {dataframe_name: file_path}
          `dataframe_name` will be the key to use to get access to a given DataFrame in
          the returned dict.
        file_format: The file format, can be either "csv" or "orc".
        sep: Separator character, in case `file_format` is "csv".
        infer_schema: If true, spark will infer types, in case `file_format` is "csv".

    Returns:
        A dictionary of datasets as pyspark DataFrame objects.

    """
    read_options = (
        {"inferSchema": infer_schema, "header": True, "sep": sep}
        if file_format == "csv"
        else {}
    )
    datasets: Dict[str, pyspark.sql.DataFrame] = {}

    spark = sf_datalake.utils.get_spark_session()
    for name, file_path in data_paths.items():
        if file_format in ("csv", "orc"):
            df = spark.read.format(file_format).options(**read_options).load(file_path)
        else:
            raise ValueError(f"Unknown file format {file_format}.")
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
    sf_datalake.transform.vector_disassembler(
        test_data,
        ["comp_probability", "probability"],
        assembled_col="probability",
        keep=["siren", "failure"],
    ).select(["siren", "failure", "probability"]).repartition(n_rep).write.csv(
        test_output_path, header=True
    )

    logging.info("Writing prediction data to file %s", prediction_output_path)
    sf_datalake.transform.vector_disassembler(
        prediction_data,
        ["comp_probability", "probability"],
        assembled_col="probability",
        keep=["siren"],
    ).select(["siren", "probability"]).repartition(n_rep).write.csv(
        prediction_output_path, header=True
    )


def write_explanations(
    output_dir: str,
    macro_scores_df: pyspark.sql.DataFrame,
    concerning_scores_df: pyspark.sql.DataFrame,
    n_rep: int = 5,
):
    """Writes the explanations of a prediction to CSV files."""
    concerning_output_path = path.join(output_dir, "concerning_values.csv")
    explanation_output_path = path.join(output_dir, "explanation_data.csv")
    logging.info("Writing concerning features to file %s", concerning_output_path)
    concerning_scores_df.repartition(n_rep).write.csv(
        concerning_output_path, header=True
    )

    logging.info(
        "Writing explanation macro scores data to directory %s", explanation_output_path
    )
    macro_scores_df.repartition(n_rep).write.csv(
        path.join(explanation_output_path), header=True
    )
