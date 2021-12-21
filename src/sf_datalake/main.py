"""Main script for statistical prediction of company failure."""

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
from sf_datalake.config import get_config
from sf_datalake.io import load_data, write_output_model
from sf_datalake.sampler import sample_df
from sf_datalake.transformer import FormatProbability
from sf_datalake.utils import instantiate_spark_session

config = get_config("configs/config_base.json")  # [TODO] - Need to adjust the path here
preprocessor = getattr(sf_datalake.preprocessor, config["PREPROCESSOR"])(config)
spark = instantiate_spark_session()

# Prepare data

indics_annuels = load_data(
    {
        "indics_annuels": path.join(
            config["DATA_ROOT_DIR"], "base/indicateurs_annuels.orc"
        )
    }
)["indics_annuels"]

if config["FILL_MISSING_VALUES"]:
    logging.info("Filling missing values with default values.")
logging.info("Aggregating data at the SIREN level")
logging.info("Feature engineering")
logging.info("Creating objective variable 'failure_within_18m'")
logging.info("Filtering out firms on 'effectif' and 'code_naf' variables.")
indics_annuels = preprocessor.run(indics_annuels)

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
stages += [sf_datalake.model.generate_stage(config)]
stages += [FormatProbability()]

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
