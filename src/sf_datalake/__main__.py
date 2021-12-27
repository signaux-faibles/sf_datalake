"""Main script for statistical prediction of company failure."""

import argparse
import logging
import os
import sys
from os import path

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413

from pyspark.ml import Pipeline

import sf_datalake.model
import sf_datalake.preprocessor
import sf_datalake.transformer
import sf_datalake.utils
from sf_datalake.io import load_data, write_output_model
from sf_datalake.sampler import sample_df

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file name (including '.json' extension).",
    default="base.json",
)
args = parser.parse_args()

config = sf_datalake.utils.get_config(args.config)
spark = sf_datalake.utils.instantiate_spark_session()

# Prepare data

indics_annuels = load_data(
    {
        "indics_annuels": path.join(
            config["DATA_ROOT_DIR"], "base/indicateurs_annuels.orc"
        )
    }
)["indics_annuels"]

pipeline_preprocessor = Pipeline(
    stages=sf_datalake.preprocessor.generate_stages(config)
)
indics_annuels = pipeline_preprocessor.fit(indics_annuels).transform(indics_annuels)

logging.info(
    "Creating oversampled training set with positive examples ratio %.1f",
    config["OVERSAMPLING_RATIO"],
)
logging.info("Creating train between %s and %s.", *config["TRAIN_DATES"])
logging.info("Creating test set between %s and %s.", *config["TEST_DATES"])
logging.info("Creating a prediction set on %s.", config["PREDICTION_DATE"])
data_train, data_test, data_prediction = sample_df(indics_annuels, config)

# Build and run Pipeline

logging.info(
    "Training %s \
    %.3f and %d iterations (maximum).",
    config["MODEL"]["MODEL_NAME"],
    config["MODEL"]["REGULARIZATION_COEFF"],
    config["MODEL"]["MAX_ITER"],
)  # [TODO] Create an attribute as an array in the config file that
# list all the parameters related to the model. Then adjust the logging to be more generic

stages = []
stages += sf_datalake.transformer.generate_stages(config)
stages += sf_datalake.model.generate_stages(config)
stages += [sf_datalake.transformer.ProbabilityFormatter()]

pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(data_train)
data_train_transformed = pipeline_model.transform(data_train)
data_test_transformed = pipeline_model.transform(data_test)
data_prediction_transformed = pipeline_model.transform(data_prediction)

model = pipeline_model.stages[-2]

logging.info(
    "Model weights: %.3f", model.coefficients
)  # [TODO] Find a more generic way, what if model is not parametric
logging.info(
    "Model intercept: %.3f", model.intercept
)  # [TODO] Find a more generic way, what if model is not parametric

macro_scores_df, micro_scores_df = sf_datalake.model.explain(
    config, model, data_prediction_transformed
)

write_output_model(
    config["OUTPUT_ROOT_DIR"],
    data_test_transformed,
    data_prediction_transformed,
    macro_scores_df,
    micro_scores_df,
)
