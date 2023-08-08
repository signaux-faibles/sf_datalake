"""Main script for statistical prediction of company failure.

Processes datasets according to provided configuration to make predictions.
"""

import argparse
import datetime as dt
import logging
import os
import random
import sys
from os import path

import numpy as np
import pyspark
from pyspark.ml import Pipeline, PipelineModel

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

# pylint: disable=C0413

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
    "--parameters",
    help="""
    Parameters file name (including '.json' extension). If not provided,
    'standard.json' will be used.
    """,
    default="standard.json",
)
parser.add_argument(
    "--variables",
    help="""
    File name (including '.json' extension) containing variables and features to
    use in the run as well as default values and transformations to be applied on
    features. If not provided, 'standard.json' will be used.
    """,
    default="standard.json",
)
parser.add_argument(
    "--dataset",
    dest="DATASET",
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
    results. If a key cannot be found inside the configuration, it will be silently
    ignored.
    """,
)

args = parser.parse_args()

# Parse configuration files and possibly override parameters.
# Then, dump all used configuration inside the output directory.
parameters = sf_datalake.io.load_parameters(args.parameters)
variables = sf_datalake.io.load_variables(args.variables)
config = {**parameters, **variables}
override_args = {k: v for k, v in vars(args).items() if k in config and v is not None}
for param, value in override_args.items():
    config[param] = value
if args.output_directory is None:
    output_directory = path.join(
        config["OUTPUT_ROOT_DIR"], str(int(dt.datetime.now().timestamp()))
    )
else:
    output_directory = args.output_directory
config["SEED"] = random.randint(0, 10000) if args.SEED is None else args.SEED
config["TRANSFORMER_FEATURES"] = {}
for feature, transformer in config["FEATURES"].items():
    config["TRANSFORMER_FEATURES"].setdefault(transformer, []).append(feature)
sf_datalake.io.dump_configuration(output_directory, config, args.dump_keys)

# Prepare data.
dataset = sf_datalake.io.load_data(
    {"dataset": config["DATASET"]},
    file_format="orc",
    spl_ratio=config["SAMPLE_RATIO"],
    seed=config["SEED"],
)["dataset"]


# Switches
with_paydex = False
if {"paydex_bin", "paydex_nb_jours_diff12m"} & set(config["FEATURES"]):
    with_paydex = True
    logging.info(
        "Paydex data features were requested through the provided configuration file. \
        The dataset will be filtered to only keep samples with available paydex data."
    )

## Pre-processing pipeline

filter_steps = []
if with_paydex:
    filter_steps.append(sf_datalake.transform.HasPaydexFilter())
normalizing_steps = [
    sf_datalake.transform.TimeNormalizer(
        inputCols=config["FEATURE_GROUPS"]["sante_financiere"],
        start="date_deb_exercice",
        end="date_fin_exercice",
    ),
    # sf_datalake.transform.TimeNormalizer(
    #     inputCols=[""], start="date_deb_tva", end="date_fin_tva"
    # ),
]
feature_engineering_steps = []
if with_paydex:
    feature_engineering_steps.append(
        sf_datalake.transform.PaydexOneHotEncoder(
            bins=config["ONE_HOT_CATEGORIES"]["paydex_bin"]
        ),
    )
    # Add corresponding 'meso' column names to the configuration for explanation step.
    config["MESO_GROUPS"]["paydex_bin"] = [
        f"paydex_bin_ohcat{i}"
        for i, _ in enumerate(config["ONE_HOT_CATEGORIES"]["paydex_bin"])
    ]

building_steps = [
    sf_datalake.transform.TargetVariable(
        inputCol=config["TARGET"]["judgment_date_col"],
        outputCol=config["TARGET"]["class_col"],
        n_months=config["TARGET"]["n_months"],
    ),
    sf_datalake.transform.ColumnSelector(
        inputCols=(
            config["IDENTIFIERS"]
            + list(config["FEATURES"])  # features dict keys to list
            + [config["TARGET"]["class_col"]]  # contains a single string
        )
    ),
    sf_datalake.transform.MissingValuesHandler(
        inputCols=list(config["FEATURES"]),
        fill=config["FILL_MISSING_VALUES"],
        value=config["DEFAULT_VALUES"],
    ),
]
preprocessing_pipeline = PipelineModel(
    stages=filter_steps + normalizing_steps + feature_engineering_steps + building_steps
)
dataset = preprocessing_pipeline.transform(dataset).cache()

# Split the dataset into train, test, predict subsets.
(
    train_data,
    test_data,
    prediction_data,
) = sf_datalake.sampler.train_test_predict_split(dataset, config)

# Build and run Pipeline
transforming_stages = sf_datalake.transform.generate_transforming_stages(config)
model_stages = [
    sf_datalake.model.get_model_from_conf(
        config["MODEL"], target_col=config["TARGET"]["class_col"]
    )
]

pipeline = Pipeline(stages=transforming_stages + model_stages)
pipeline_model = pipeline.fit(train_data)
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)
prediction_transformed = pipeline_model.transform(prediction_data)

# Explain predictions
model = sf_datalake.model.get_model_from_pipeline_model(
    pipeline_model, config["MODEL"]["NAME"]
)
if isinstance(model, pyspark.ml.classification.LogisticRegressionModel):
    logging.info("Model weights: %.3f", model.coefficients)
    logging.info("Model intercept: %.3f", model.intercept)

features_list = sf_datalake.utils.feature_index(config)
shap_values, expected_value = sf_datalake.explain.explanation_data(
    features_list, model, train_transformed, prediction_transformed
)
macro_scores, concerning_scores = sf_datalake.explain.explanation_scores(
    config, shap_values
)
# Convert to [0, 1] range if shap values are expressed in log-odds units.
if isinstance(
    model,
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
    output_directory,
    test_transformed,
    prediction_transformed,
)
sf_datalake.io.write_explanations(
    output_directory,
    spark.createDataFrame(macro_scores.reset_index()),
    spark.createDataFrame(concerning_scores.reset_index()),
)
