"""Main script for statistical prediction of company failure.

Processes datasets according to provided configuration to make predictions.
"""
# pylint: disable=unsubscriptable-object,wrong-import-position

import argparse
import logging
import os
import sys
from os import path
from typing import List

import numpy as np
import pyspark
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

# isort: off
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/"))
sys.path.append(path.join(os.getcwd(), "venv/lib/python3.6/site-packages/"))
# isort: on

import sf_datalake.configuration
import sf_datalake.explain
import sf_datalake.io
import sf_datalake.model_selection
import sf_datalake.transform
import sf_datalake.utils

spark = sf_datalake.utils.get_spark_session()
parser = argparse.ArgumentParser(
    description="""
    Run a 'Signaux Faibles' distributed prediction with the chosen set of
    parameters and variables.
    """
)
path_group = parser.add_argument_group(
    "paths", description="Path command line arguments."
)
parser.add_argument(
    "--configuration",
    help="""
    Configuration file name (including '.json' extension). If not provided,
    'standard.json' will be used.
    """,
    default="standard.json",
)
path_group.add_argument(
    "--root_directory",
    type=str,
    help="Data root directory.",
)
path_group.add_argument(
    "--dataset",
    dest="dataset_path",
    type=str,
    help="""
    Path (relative to root_directory) to the dataset that will be used for training,
    test or prediction.""",
)
path_group.add_argument(
    "--prediction_path",
    type=str,
    help="""
    Path (relative to root_directory) where predictions and parameters will be saved.
    """,
)
parser.add_argument(
    "--train_dates",
    type=str,
    nargs=2,
    help="The training set start and end dates (YYYY-MM-DD format).",
)
parser.add_argument(
    "--prediction_date",
    type=str,
    help="The date over which prediction should be made (YYYY-MM-DD format).",
)
parser.add_argument(
    "--sample_ratio",
    type=float,
    help="Loaded data sample size as a fraction of its full size.",
)
parser.add_argument(
    "--drop_missing_values",
    action="store_true",
    help="""
    If specified, missing values will be dropped instead of filling data with default
    values.
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
    {
        "dataset": path.join(
            configuration.io.root_directory, configuration.io.dataset_path
        )
    },
    file_format="orc",
).popitem()

if configuration.io.sample_ratio != 1.0:
    raw_dataset = raw_dataset.sample(
        fraction=configuration.io.sample_ratio, seed=configuration.io.random_seed
    )


## Pre-processing pipeline
filter_steps = []
if not configuration.preprocessing.drop_missing_values:
    raise NotImplementedError(
        " VectorAssembler in spark < 2.4.0 doesn't handle including missing values."
    )
filter_steps.append(
    sf_datalake.transform.MissingValuesDropper(inputCols=raw_dataset.columns)
)

preprocessing_pipeline = Pipeline(
    stages=filter_steps + configuration.encoding_scaling_stages()
)

preprocessing_pipeline_model = preprocessing_pipeline.fit(raw_dataset)
pre_dataset = preprocessing_pipeline_model.transform(raw_dataset).cache()

# Split the dataset into train, test for evaluation.
train_data, test_data = sf_datalake.model_selection.train_test_split(
    pre_dataset.filter(
        (
            sf_datalake.utils.to_date(configuration.learning.train_dates[0])
            <= F.col("période")
        )
        & (
            F.col("période")
            < sf_datalake.utils.to_date(configuration.learning.train_dates[1])
        )
    ),
    configuration.io.random_seed,
    train_size=configuration.learning.train_size,
    group_col="siren",
)
prediction_data = pre_dataset.filter(
    F.col("période")
    == sf_datalake.utils.to_date(configuration.learning.prediction_date)
)

# Resample train dataset following requested classes balance
resampler = sf_datalake.transform.RandomResampler(
    class_col=configuration.learning.target["class_col"],
    method=configuration.learning.target["resampling_method"],
    min_class_ratio=configuration.learning.target["target_resampling_ratio"],
    seed=configuration.io.random_seed,
)
resampled_train_data = resampler.transform(train_data)

# Fit ML model and make predictions
classifier = configuration.learning.get_model()
classifier_model = classifier.fit(resampled_train_data)
train_transformed = classifier_model.transform(resampled_train_data)
test_transformed = classifier_model.transform(test_data)
prediction_transformed = classifier_model.transform(prediction_data)


# TODO: Update this for other models
if isinstance(classifier_model, pyspark.ml.classification.LogisticRegressionModel):
    logging.info("Model weights: %.3f", classifier_model.coefficients)
    logging.info("Model intercept: %.3f", classifier_model.intercept)

# Get features names
model_features: List[str] = sf_datalake.utils.extract_column_names(
    pre_dataset, configuration.learning.features_column
)

for scaler in list(configuration.preprocessing.scalers_params):
    scaler_columns = sf_datalake.utils.extract_column_names(
        pre_dataset, f"{scaler}_input"
    )
    for i, col in enumerate(model_features):
        if col.startswith(scaler):
            model_features[i] = scaler_columns[int(col.split("_")[-1])]

# Compute predictions explanation
shap_values, expected_value = sf_datalake.explain.explanation_data(
    model_features,
    configuration.learning.features_column,
    classifier_model,
    train_transformed,
    prediction_transformed,
    configuration.explanation.n_train_sample,
)
macro_scores, concerning_scores = sf_datalake.explain.explanation_scores(
    shap_values,
    configuration.explanation.topic_groups,
    configuration.explanation.n_concerning_micro,
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
    path.join(configuration.io.root_directory, configuration.io.prediction_path),
    test_transformed,
    prediction_transformed,
)
sf_datalake.io.write_explanations(
    path.join(configuration.io.root_directory, configuration.io.prediction_path),
    spark.createDataFrame(macro_scores.reset_index()),
    spark.createDataFrame(concerning_scores.reset_index()),
)
