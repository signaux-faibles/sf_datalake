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
    output_format: str,
    n_rep: int = 5,
):
    """Writes the results of a prediction to CSV or parquet files."""
    test_output_path = path.join(output_dir, "test_data." + output_format)
    prediction_output_path = path.join(output_dir, "prediction_data." + output_format)

    output_test = (
        sf_datalake.transform.vector_disassembler(
            test_data,
            ["comp_probability", "probability"],
            assembled_col="probability",
            keep=["siren", "failure", "code_naf", "code_commune", "région"],
        )
        .select(
            ["siren", "code_naf", "failure", "probability", "code_commune", "région"]
        )
        .repartition(n_rep)
    )

    output_prediction = (
        sf_datalake.transform.vector_disassembler(
            prediction_data,
            ["comp_probability", "probability"],
            assembled_col="probability",
            keep=["siren", "code_naf", "code_commune", "région"],
        )
        .select(["siren", "code_naf", "probability", "code_commune", "région"])
        .repartition(n_rep)
    )

    if output_format == "csv":
        logging.info("Writing test data to file %s", test_output_path)
        output_test.write.csv(test_output_path, header=True)
        logging.info("Writing prediction data to file %s", prediction_output_path)
        output_prediction.write.csv(prediction_output_path, header=True)
    elif output_format == "parquet":
        logging.info("Writing test data to file %s", test_output_path)
        output_test.coalesce(1).write.parquet(test_output_path)
        logging.info("Writing prediction data to file %s", prediction_output_path)
        output_prediction.coalesce(1).write.parquet(prediction_output_path)
    else:
        raise ValueError(f"Unknown file format {output_format}.")


def write_explanations(
    output_dir: str,
    macro_scores_df: pyspark.sql.DataFrame,
    micro_scores_df: pyspark.sql.DataFrame,
    output_format: str,
    n_rep: int = 5,
):
    """Writes the explanations of a prediction to CSV or parquet files."""
    micro_output_path = path.join(output_dir, "micro_explanation." + output_format)
    macro_output_path = path.join(output_dir, "macro_explanation." + output_format)

    if output_format == "csv":
        logging.info("Writing micro explanation data to file %s", micro_output_path)
        micro_scores_df.repartition(n_rep).write.csv(micro_output_path, header=True)
        logging.info(
            "Writing macro explanation data to directory %s", macro_output_path
        )
        macro_scores_df.repartition(n_rep).write.csv(macro_output_path, header=True)
    elif output_format == "parquet":
        logging.info("Writing micro explanation data to file %s", micro_output_path)
        micro_scores_df.coalesce(1).write.parquet(micro_output_path)
        logging.info(
            "Writing macro explanation data to directory %s", macro_output_path
        )
        macro_scores_df.coalesce(1).write.parquet(macro_output_path)
    else:
        raise ValueError(f"Unknown file format {output_format}.")
