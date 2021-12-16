"""Logistic regression model for company failure prediction.
"""
import os
import sys
from os import path

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413

from sf_datalake.config import Config
from sf_datalake.io import instantiate_spark_session, load_data, write_output_model
from sf_datalake.model import factory_model
from sf_datalake.preprocessor import factory_preprocessor
from sf_datalake.sampler import factory_sampler
from sf_datalake.transformer import factory_transformer

# Get Pipeline objects
config = Config("configs/config_base.json")  # [TODO] - Need to adjust the path here
preprocessor = factory_preprocessor(config)
sampler = factory_sampler(config)
transformer = factory_transformer(config)
model = factory_model(config)

# Run the Pipeline
spark = instantiate_spark_session()

indics_annuels = load_data(
    {
        "indics_annuels": path.join(
            config.get_config()["DATA_ROOT_DIR"], "base/indicateurs_annuels.orc"
        )
    }
)["indics_annuels"]

indics_annuels = preprocessor.run(indics_annuels)

data_train, data_test, data_prediction = sampler.run(indics_annuels)

scaled_train, scaled_test, scaled_prediction = transformer.run(
    data_train, data_test, data_prediction
)

test_data = model.fit(scaled_train, scaled_test)
prediction_data = model.predict(scaled_prediction)

macro_scores_df, micro_scores_df = model.explain(scaled_prediction)

write_output_model(
    config.get_config()["OUTPUT_ROOT_DIR"],
    test_data,
    prediction_data,
    macro_scores_df,
    micro_scores_df,
)
