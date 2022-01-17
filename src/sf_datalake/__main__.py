"""Main script for statistical prediction of company failure."""

import argparse
import datetime
import logging
import os
import random
import sys
from os import path

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413

from pyspark.ml import Pipeline

import sf_datalake.io
import sf_datalake.model
import sf_datalake.sampler
import sf_datalake.transform
import sf_datalake.utils


def main(args: argparse.Namespace):  # pylint: disable=R0914
    """Processes datasets according to configuration to make predictions."""

    # Parse a configuration file and possibly override parameters.
    config = sf_datalake.utils.get_config(args.configuration)
    config_args = {k: v for k, v in vars(args).items() if k in config and v is not None}
    for param, value in config_args.items():
        config[param] = value
    if args.output_directory is None:
        output_directory = path.join(
            config["OUTPUT_ROOT_DIR"],
            "sorties_modeles",
            datetime.date.today().isoformat(),
        )
    else:
        output_directory = args.output_directory
    config["SEED"] = random.randint(0, 10000) if args.SEED is None else args.SEED
    sf_datalake.io.dump_configuration(output_directory, config, args.dump_keys)

    # Prepare data.
    yearly_data = sf_datalake.io.load_data(
        {
            "yearly_data": path.join(
                config["DATA_ROOT_DIR"], "base/indicateurs_annuels.orc"
            ),
        },
        spl_ratio=config["SAMPLE_RATIO"],
        seed=config["SEED"],
    )["yearly_data"]

    pipeline_preprocessor = Pipeline(
        stages=sf_datalake.transform.generate_preprocessing_stages(config)
    )
    yearly_data = pipeline_preprocessor.fit(yearly_data).transform(yearly_data)

    logging.info(
        "Creating oversampled training set with positive examples ratio %.1f",
        config["TARGET_OVERSAMPLING_RATIO"],
    )
    logging.info("Creating train between %s and %s.", *config["TRAIN_DATES"])
    logging.info("Creating test set between %s and %s.", *config["TEST_DATES"])
    logging.info("Creating a prediction set on %s.", config["PREDICTION_DATE"])
    (
        train_data,
        test_data,
        prediction_data,
    ) = sf_datalake.sampler.train_test_predict_split(yearly_data, config)

    # Build and run Pipeline
    logging.info(
        "Training %s \
        %.3f and %d iterations (maximum).",
        config["MODEL"]["NAME"],
        config["MODEL"]["REGULARIZATION_COEFF"],
        config["MODEL"]["MAX_ITER"],
    )

    scaling_stages = sf_datalake.transform.generate_scaling_stages(config)
    model_stages = sf_datalake.model.generate_stages(config)
    postprocessing_stages = [sf_datalake.transform.ProbabilityFormatter()]

    pipeline = Pipeline(stages=scaling_stages + model_stages + postprocessing_stages)
    pipeline_model = pipeline.fit(train_data)
    _ = pipeline_model.transform(train_data)
    model = sf_datalake.model.get_model_from_pipeline_model(
        pipeline_model, config["MODEL"]["NAME"]
    )
    logging.info("Model weights: %.3f", model.coefficients)
    logging.info("Model intercept: %.3f", model.intercept)
    test_transformed = pipeline_model.transform(test_data)
    prediction_transformed = pipeline_model.transform(prediction_data)
    macro_scores, micro_scores = sf_datalake.model.explain(
        config, pipeline_model, prediction_transformed
    )

    # Write outputs.
    sf_datalake.io.write_predictions(
        output_directory,
        test_transformed,
        prediction_transformed,
    )
    sf_datalake.io.write_explanations(
        output_directory,
        macro_scores,
        micro_scores,
    )


if __name__ == "__main__":
    _ = sf_datalake.utils.get_spark_session()
    parser = argparse.ArgumentParser(
        description="""
        Run a 'Signaux Faibles' distributed prediction with the chosen set of
        parameters.
        """
    )
    parser.add_argument(
        "--configuration",
        help="""
        Configuration file name (including '.json' extension). If not provided,
        'base.json' will be used.
        """,
        default="base.json",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Directory where model predictions and parameters will be saved.",
    )
    parser.add_argument(
        "--train_dates",
        dest="TRAIN_DATES",
        type=str,
        nargs=2,
        help="The training set start and end dates (YYYY-MM-DD format).",
    )
    parser.add_argument(
        "--test_dates",
        dest="TEST_DATES",
        type=str,
        nargs=2,
        help="The test set start and end dates (YYYY-MM-DD format).",
    )
    parser.add_argument(
        "--prediction_date",
        dest="PREDICTION_DATE",
        type=str,
        help="The date over which prediction should be made (YYYY-MM-DD format).",
    )
    parser.add_argument(
        "--sample_ratio",
        dest="SAMPLE_RATIO",
        type=float,
        help="The loaded data sample size as a fraction of its complete size.",
    )
    parser.add_argument(
        "--oversampling",
        dest="TARGET_OVERSAMPLING_RATIO",
        type=float,
        help="""
        Enforces the ratio of positive observations ("entreprises en défaillance") to be
        the specified ratio.
        """,
    )
    parser.add_argument(
        "--drop_missing_values",
        dest="FILL_MISSING_VALUES",
        action="store_false",
        help="""
        If specified, missing values will be dropped instead of filling data with
        default values.
        """,
    )
    parser.add_argument(
        "--seed",
        dest="SEED",
        type=int,
        help="""
        If specified, the seed used in all calls of the following functions: pyspark.sql.DataFrame.sample(), pyspark.sql.DataFrame.randomSplit(). If not specified, a random value is used.
        """,
    )
    parser.add_argument(
        "--dump_keys",
        type=str,
        nargs="+",
        help="""
        A sequence of configuration keys that should be dumped along with the prediction
        results. If a key cannot be found inside the configuration, it will be silently
        ignored.
        """,
    )
    main(parser.parse_args())
