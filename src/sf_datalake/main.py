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

import sf_datalake.model
import sf_datalake.preprocessor
import sf_datalake.transformer
from sf_datalake.config import get_config
from sf_datalake.io import load_data, write_output_model
from sf_datalake.sampler import sample_df
from sf_datalake.utils import instantiate_spark_session

# Get Pipeline objects
config = get_config("configs/config_base.json")  # [TODO] - Need to adjust the path here
preprocessor = getattr(sf_datalake.preprocessor, config["PREPROCESSOR"])(config)
transformer = getattr(sf_datalake.transformer, config["TRANSFORMER"])(config)
model = getattr(sf_datalake.model, config["MODEL"])(config)

# Run the Pipeline
spark = instantiate_spark_session()

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

transformer.fit(data_train)
scaled_train = transformer.transform(
    df=data_train,
    label_col="failure_within_18m",
)
scaled_test = transformer.transform(
    df=data_test,
    label_col="failure_within_18m",
    keep_cols=["siren", "time_til_failure"],
)
scaled_prediction = transformer.transform(
    df=data_prediction,
    label_col="failure_within_18m",
    keep_cols=["siren"],
)

logging.info(
    "Training %s \
    %.3f and %d iterations (maximum).",
    config["MODEL"],
    config["REGULARIZATION_COEFF"],
    config["MAX_ITER"],
)  # [TODO] Create an attribute as an array in the config file that
# list all the parameters related to the model. Then adjust the logging to be more generic

model.fit(scaled_train)
logging.info(
    "Model weights: %.3f", model.model.coefficients
)  # [TODO] Find a more generic way, what if model is not parametric
logging.info(
    "Model intercept: %.3f", model.model.intercept
)  # [TODO] Find a more generic way, what if model is not parametric

logging.info("Running model on test dataset.")
test_data = model.predict(scaled_test)

logging.info("Running model on prediction dataset.")
prediction_data = model.predict(scaled_prediction)

macro_scores_df, micro_scores_df = model.explain(scaled_prediction)

write_output_model(
    config["OUTPUT_ROOT_DIR"],
    test_data,
    prediction_data,
    macro_scores_df,
    micro_scores_df,
)
