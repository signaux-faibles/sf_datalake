"""Main script for statistical prediction of company failure.

Processes datasets according to provided configuration to make predictions.
"""
# pylint: disable=unsubscriptable-object,wrong-import-position

import argparse
import logging
import os
import sys
from os import path
from typing import Dict, List

import numpy as np
import pyspark
from pyspark.ml import Pipeline

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

import sf_datalake.configuration
import sf_datalake.explain
import sf_datalake.io
import sf_datalake.model
import sf_datalake.sampler
import sf_datalake.transform
import sf_datalake.utils

spark = sf_datalake.utils.get_spark_session()
parser = argparse.ArgumentParser(
    description="""
    Run a 'Signaux Faibles' distributed prediction with the chosen set of
    parameters and variables.
    """
)
parser.add_argument(
    "--configuration",
    help="""
    Configuration file name (including '.json' extension). If not provided,
    'standard.json' will be used.
    """,
    default="standard.json",
)

parser.add_argument(
    "--dataset",
    dest="dataset_path",
    type=str,
    help="Path to the dataset that will be used for training, test and prediction.",
)
parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory where model predictions and parameters will be saved.",
)
parser.add_argument(
    "--train_dates",
    type=str,
    nargs=2,
    help="The training set start and end dates (YYYY-MM-DD format).",
)
parser.add_argument(
    "--test_dates",
    type=str,
    nargs=2,
    help="The test set start and end dates (YYYY-MM-DD format).",
)
parser.add_argument(
    "--prediction_date",
    type=str,
    help="The date over which prediction should be made (YYYY-MM-DD format).",
)
parser.add_argument(
    "--sample_ratio",
    type=float,
    help="The loaded data sample size as a fraction of its complete size.",
)
parser.add_argument(
    "--drop_missing_values",
    action="store_true",
    help="""
    If specified, missing values will be dropped instead of filling data with
    default values.
    """,
)
parser.add_argument(
    "--seed",
    dest="random_seed",
    type=int,
    help="""
    If specified, the seed used in all calls of the following functions:
    pyspark.sql.DataFrame.sample(), pyspark.sql.DataFrame.randomSplit(). If not
    specified, a random value is used.
    """,
)
parser.add_argument(
    "--dump_keys",
    type=str,
    nargs="+",
    help="""
    A sequence of configuration keys that should be dumped along with the prediction
    results.
    """,
)

args = vars(parser.parse_args())

# Parse configuration files and possibly override parameters.
# Then, dump all used configuration inside the output directory.
config_file: str = args.pop("configuration")
dump_keys: List[str] = args.pop("dump_keys")

configuration = sf_datalake.configuration.ConfigurationHelper(
    config_file=config_file, cli_args=args
)
configuration.dump(dump_keys)

# Prepare data.
_, raw_dataset = sf_datalake.io.load_data(
    {"dataset": configuration.io.dataset_path},
    file_format="orc",
).popitem()

if configuration.io.sample_ratio != 1.0:
    raw_dataset = raw_dataset.sample(
        fraction=configuration.io.sample_ratio, seed=configuration.io.random_seed
    )


# Switches
with_paydex = False
if any(
    "paydex" in feat for feat in set(configuration.preprocessing.features_transformers)
):
    with_paydex = True
    logging.info(
        "Paydex data features were requested through the provided configuration file. \
        The dataset will be filtered to only keep samples with available paydex data."
    )

## Pre-processing pipeline

filter_steps = []
if with_paydex:
    filter_steps.append(sf_datalake.transform.HasPaydexFilter())

# TODO: this only concerns data from a particular source and should be
# handled before the main script.
normalizing_steps = [
    sf_datalake.transform.TimeNormalizer(
        inputCols=configuration.explanation.topic_groups["sante_financiere"],
        start="date_deb_exercice",
        end="date_fin_exercice",
    ),
    # sf_datalake.transform.TimeNormalizer(
    #     inputCols=[""], start="date_deb_tva", end="date_fin_tva"
    # ),
]

building_steps = [
    sf_datalake.transform.TargetVariable(
        inputCol=configuration.learning.target["judgment_date_col"],
        outputCol=configuration.learning.target["class_col"],
        n_months=configuration.learning.target["n_months"],
    ),
]


missing_values_handling_steps = []
if configuration.preprocessing.fill_default_values:
    missing_values_handling_steps.append(
        sf_datalake.transform.MissingValuesHandler(
            inputCols=list(configuration.preprocessing.fill_default_values),
            value=configuration.preprocessing.fill_default_values,
        ),
    )
if configuration.preprocessing.fill_imputation_strategy:
    imputation_strategy_features: Dict[str, List[str]] = {}
    for (
        feature,
        strategy,
    ) in configuration.preprocessing.fill_imputation_strategy.items():
        imputation_strategy_features.setdefault(strategy, []).append(feature)
    missing_values_handling_steps.extend(
        sf_datalake.transform.MissingValuesHandler(
            inputCols=features,
            stat_strategy=strategy,
        )
        for strategy, features in imputation_strategy_features.items()
    )


preprocessing_pipeline = Pipeline(
    stages=filter_steps
    + normalizing_steps
    + building_steps
    + missing_values_handling_steps
    + configuration.transforming_stages()
)

pre_dataset = preprocessing_pipeline.transform(raw_dataset)
if configuration.preprocessing.drop_missing_values:
    pre_dataset = pre_dataset.dropna()
pre_dataset = pre_dataset.cache()

# Split the dataset into train, test, predict subsets.
(
    train_data,
    test_data,
    prediction_data,
) = sf_datalake.sampler.train_test_predict_split(
    pre_dataset,
    configuration.learning.target["class_col"],
    configuration.learning.target["oversampling_ratio"],
    configuration.learning.train_test_split_ratio,
    configuration.learning.train_dates,
    configuration.learning.test_dates,
    configuration.learning.prediction_date,
    configuration.io.random_seed,
)

# Fit ML model and make predictions

classifier = configuration.learning.get_model()
classifier_model = classifier.fit(train_data)
train_transformed = classifier_model.transform(train_data)
test_transformed = classifier_model.transform(test_data)
prediction_transformed = classifier_model.transform(prediction_data)


# TODO: Update this for other models
if isinstance(classifier_model, pyspark.ml.classification.LogisticRegressionModel):
    logging.info("Model weights: %.3f", classifier_model.coefficients)
    logging.info("Model intercept: %.3f", classifier_model.intercept)

# TODO:
# - Adapt MESO list using e.g. regex
features_metadata = prediction_data.schema[
    configuration.learning.feature_column
].metadata["ml_attr"]
model_features: List[str] = [None] * features_metadata["num_attrs"]

for var_type, variables in features_metadata["attrs"].items():
    for variable_dict in variables:
        model_features[variable_dict["idx"]] = model_features[variable_dict["name"]]


shap_values, expected_value = sf_datalake.explain.explanation_data(
    model_features,
    classifier_model,
    train_transformed,
    prediction_transformed,
    configuration.explanation.n_train_sample,
)
macro_scores, concerning_scores = sf_datalake.explain.explanation_scores(
    shap_values,
    configuration.explanation.n_concerning_micro,
    configuration.explanation.topic_groups,
)
# Convert to [0, 1] range if shap values are expressed in log-odds units.
if isinstance(
    classifier_model,
    (
        pyspark.ml.classification.LogisticRegressionModel,
        pyspark.ml.classification.GBTClassificationModel,
    ),
):
    num_cols = concerning_scores.select_dtypes(include="number").columns
    concerning_scores.loc[:, num_cols] = 1 / (1 + np.exp(-concerning_scores[num_cols]))
    macro_scores = 1 / (1 + np.exp(-macro_scores))

# Write outputs.
sf_datalake.io.write_predictions(
    configuration.io.output_directory,
    test_transformed,
    prediction_transformed,
)
sf_datalake.io.write_explanations(
    configuration.io.output_directory,
    spark.createDataFrame(macro_scores.reset_index()),
    spark.createDataFrame(concerning_scores.reset_index()),
)
