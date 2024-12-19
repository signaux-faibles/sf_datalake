# pylint: disable=unsubscriptable-object
"""Configuration helper classes."""

import dataclasses
import datetime as dt
import inspect
import json
import random
import shutil
import time
from dataclasses import dataclass
from os import path, remove
from typing import Any, Dict, Iterable, List, Tuple

import importlib_metadata
import importlib_resources
from pyspark.ml import Estimator, Transformer
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)

import sf_datalake.utils
from sf_datalake.transform import BinsOrdinalEncoder


def extract_dc_fields(dcls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract some arguments to provide to some dataclass constructor."""
    class_fields = set(field.name for field in dataclasses.fields(dcls))
    extracted = {}
    for k in class_fields:
        if k in kwargs:
            extracted[k] = kwargs.pop(k)
    return {k: v for k, v in extracted.items() if v is not None}


@dataclass
class LearningConfiguration:
    """Machine learning configuration.

    Attributes:
        target: A mapping containing the following :
          - "class_col": Name of the column defining the sample's class.
          - "n_months": Lengths of the window, in months, that is used to define the
            target.
          - "judgment_date_col": Name of a column containing judgment date.
          - "target_resampling_ratio": Required (target_cls / total) # of samples ratio
            used to resample the training dataset. We assume that the minority class is
            the target class inside the dataset.
          - "resampling_method": Choose between "oversampling" or "undersampling" for
            training dataset resampling.
        train_dates: Date interval (inclusive) that will be used to extract samples from
          the dataset for training.
        test_dates: Date interval (inclusive) that will be used to extract samples from
          the dataset for testing.
        prediction_date: Single month) that will be used to extract samples from the
          dataset for testing.
        train_size: Fraction of dataset to use for training vs testing.
        model_name: Name of the required model.
        model_params: Mapping from model names to mappings of these models' objects
          kwargs.
        features_column: Name of the column that will hold all features fed to the
          model.

    """

    target: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "class_col": "failure",
            "n_months": 18,
            "judgment_date_col": "date_jugement",
            "target_resampling_ratio": 0.35,
            "resampling_method": "oversampling",
        }
    )
    train_dates: Tuple[str] = ("2016-01-01", "2019-05-31")
    prediction_date: str = "2020-02-01"
    train_size: float = 0.8
    model_name: str = "LogisticRegression"
    model_params: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory=lambda: {
            "LogisticRegression": {
                "regParam": 0.12,
                "maxIter": 500,
                "tol": 1e-05,
                "standardization": False,
            },
            "RandomForestClassifier": {
                "maxDepth": 9,
                "numTrees": 100,
            },
            "GBTClassifier": {
                "maxDepth": 3,
                "maxIter": 100,
                "maxBins": 255,
                "minInstancesPerNode": 1,
                "stepSize": 0.025,  # Learning rate
            },
        }
    )
    features_column: str = "features"

    def get_model(self) -> Estimator:
        # pylint: disable=missing-function-docstring, not-a-mapping, no-member
        model_factory: Dict[str, Transformer] = {
            "LogisticRegression": LogisticRegression,
            "GBTClassifier": GBTClassifier,
            "RandomForestClassifier": RandomForestClassifier,
        }

        return (
            model_factory[self.model_name]()
            .setParams(**(self.model_params.get(self.model_name, {})))
            .setFeaturesCol(self.features_column)
            .setLabelCol(self.target["class_col"])
        )


@dataclass
class PreprocessingConfiguration:
    """Pre-processing configuration.

    Attributes:
        identifiers: Iterable of variable names that identify a given sample.
        siren_aggregation: Mapping from a variable name to an aggregation function (to
          be used when grouping by SIREN and time).
        time_aggregation: Mapping from functions to mapping from features to be
          aggregated to number of months.
        drop_missing_values: If true, drop any missing values from datasets before
          proceeding to training.
        fill_default_values: Mapping from feature name to associated default value.
        fill_imputation_strategy: Mapping from feature name to method for missing value
          imputation.
        features_transformers: Mapping from features name to iterable of transformers
          names.
        encoders_params: Encoders kwargs used for instanciation of these objects.
        ordinal_encoding_bins: Mapping from feature name to iterable of 2-uples
          representing continuous values bins for ordinal encoding such as one-hot
          encoding.
        scalers_params: Scalers kwargs used for instanciation of these objects.
    """

    identifiers: List[str] = dataclasses.field(
        default_factory=lambda: ["siren", "période"]
    )
    siren_aggregation: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "cotisation": "sum",
            "dette_sociale_ouvrière": "sum",
            "dette_sociale_patronale": "sum",
            "effectif": "sum",
            "ap_heures_consommées": "sum",
        }
    )
    # Time-series aggregates
    time_aggregation: Dict[str, Dict[str, List[int]]] = None
    # Missing values handling
    drop_missing_values: bool = True
    fill_default_values: Dict[str, Any] = None
    fill_imputation_strategy: Dict[str, Any] = None
    # Transformations
    features_transformers: Dict[str, List[str]] = None
    ordinal_encoding_bins: Dict[str, List[str]] = None
    encoders_params: Dict[str, Transformer] = dataclasses.field(
        default_factory=lambda: {
            "OneHotEncoder": {"dropLast": False},
            "StringIndexer": {},
            "BinsOrdinalEncoder": {},
        }
    )
    scalers_params: Dict[str, Transformer] = dataclasses.field(
        default_factory=lambda: {
            "StandardScaler": {
                "withMean": True,
                "withStd": True,
                "inputCol": "StandardScaler_input",
                "outputCol": "StandardScaler_output",
            },
        }
    )

    def get_scaler(self, scaler_name):  # pylint:disable=missing-function-docstring
        scaler_factory = {
            "StandardScaler": StandardScaler,
        }
        return scaler_factory[scaler_name](**self.scalers_params[scaler_name])

    def get_encoder(self, encoder_name):  # pylint:disable=missing-function-docstring
        encoder_factory = {
            "OneHotEncoder": OneHotEncoder,
            "StringIndexer": StringIndexer,
            "BinsOrdinalEncoder": BinsOrdinalEncoder,
        }
        return encoder_factory[encoder_name](**self.encoders_params[encoder_name])


@dataclass
class ExplanationConfiguration:
    """Explanation configuration.

    Attributes:
        n_train_sample: Number of training samples used for explanation case a linear
          model is used.
        topic_groups: Mapping from a topic to a list of features associated with this
          topic.

    """

    n_train_sample: int = 5000
    topic_groups: Dict[str, List[str]] = None


@dataclass
class IOConfiguration:
    """Input output configuration.

    Parameters for reading / writing paths, as well as sampling.

    Attributes:
        root_directory: Data root directory.
        dataset_path: Path (relative to root_directory) to a dataset that will be used
          for training, test or prediction.
        prediction_path: Path (relative to root_directory) where predictions and
          runtime parameters will be saved.
        sample_ratio: Loaded data sample size as a fraction of its full size.
        random_seed: An integer random seed (used during sampling operations).
        output_format : csv or parquet

    """

    root_directory: str = "/projets/TSF"
    dataset_path: str = "data/preprocessed/datasets/full_dataset"
    prediction_path: str = path.join(f"predictions/{dt.datetime.now().timestamp()}")
    sample_ratio: float = 1.0
    random_seed: int = random.randint(0, 10000)
    output_format: str = "csv"


class ConfigurationHelper:
    """Helper class for the ML procedure configuration.

    This helper will hold data described inside a configuration file or input through
    CLI, handle some transformations over this data and hand it over to other objects
    when needed.

    There are three possible sources of info for this helper, which are looked for in
    the following order:
    1) Default values of the dataclasses that are attributes of this class.
    2) Configuration file.
    3) Command line interface arguments.

    The last looked-up source will have precedence over the previous ones.

    Args:
        config_file: Basename of a config file (including its .json extension) that is
          part of this package. It will be read as a python dictionary.
        cli_args: Some mapping parsed from command-line input.

    """

    def __init__(self, config_file: str = None, cli_args: Dict[str, Any] = None):
        override_args: Dict[str, Any] = {}
        if config_file is not None:
            with importlib_resources.files("sf_datalake.configuration").joinpath(
                f"{config_file}"
            ) as f:
                override_args.update(json.loads(f.read_text()))
        if cli_args is not None:
            override_args.update(cli_args)

        self.learning = LearningConfiguration(
            **extract_dc_fields(LearningConfiguration, override_args)
        )
        self.preprocessing = PreprocessingConfiguration(
            **extract_dc_fields(PreprocessingConfiguration, override_args)
        )
        self.explanation = ExplanationConfiguration(
            **extract_dc_fields(ExplanationConfiguration, override_args)
        )
        self.io = IOConfiguration(**extract_dc_fields(IOConfiguration, override_args))
        self.version = importlib_metadata.version("sf_datalake")
        # There should not be any override argument left.
        if override_args:
            raise ValueError(
                f"Override argument(s) '{list(override_args.keys())}' could not be "
                "matched against any ConfigurationHelper attribute."
            )

        # Duplicate config for time-aggregated variables.
        def add_time_aggregate_features(attribute: dict):
            for operation in self.preprocessing.time_aggregation.keys():
                for variable, n_months in self.preprocessing.time_aggregation[
                    operation
                ].items():
                    # TODO: This condition on "diff" should be handled differently
                    if attribute.get(variable) is not None and operation != "diff":
                        attribute.update(
                            (
                                f"{variable}_{operation}{n_month}m",
                                attribute[variable],
                            )
                            for n_month in n_months
                        )

        add_time_aggregate_features(self.preprocessing.features_transformers)

    def dump(self, dump_keys: Iterable[str] = None):
        """Dumps a subset of the configuration used during a prediction run.

        Args:
            dump_keys: An Iterable of configuration parameters that should be dumped.
              All elements of `dump_keys` must be attributes of this
              ConfigurationHelper object.

        """
        spark = sf_datalake.utils.get_spark_session()

        complete_dump: Dict[str, Any] = dict(
            (attr, dataclasses.asdict(value))
            for attr, value in inspect.getmembers(
                self, predicate=dataclasses.is_dataclass
            )
        )
        complete_dump.update({"version": self.version})  # hackish...

        dump_dict = (
            {k: complete_dump[k] for k in dump_keys}
            if dump_keys is not None
            else complete_dump
        )
        config_rdd = spark.sparkContext.parallelize([json.dumps(dump_dict)])
        file_path = path.join(
            self.io.root_directory, self.io.prediction_path, "run_configuration"
        )
        if path.exists(file_path):
            shutil.rmtree(file_path)
            time.sleep(2)
        config_rdd.repartition(1).saveAsTextFile(file_path)

    def encoding_scaling_stages(self) -> List[Transformer]:
        """Generates all stages related to feature encoding and sclaing.

        Feature transformations are prepared in the following order:
        - encoding stages, which operate on single feature columns.
        - scaling stages, which operate on a set of features assembled using a
          `VectorAssembler`.

        Returns:
            The stages, ready to be used inside a pyspark.ml.Pipeline.

        """
        # pylint: disable=unsupported-membership-test
        encoding_steps: List[Transformer] = []
        scaling_steps: List[Transformer] = []
        scaler_inputs: Dict[str, List[str]] = {}

        # Features that will be eventually passed to the model will be appended along
        # the way to this list
        model_features: List[str] = []

        def is_encoder(name: str):
            return name in self.preprocessing.encoders_params

        def is_scaler(name: str):
            return name in self.preprocessing.scalers_params

        if not self.preprocessing.drop_missing_values:
            raise NotImplementedError(
                "VectorAssembler in spark < 2.4.0 doesn't handle including missing "
                "values."
            )

        # Iterate over each dataset variable, and prepare feature encoding and
        # normalizing pipeline steps.
        for (
            feature_name,
            transformer_names,
        ) in self.preprocessing.features_transformers.items():
            # Encoding
            encoders = [
                self.preprocessing.get_encoder(transformer_name)
                for transformer_name in transformer_names
                if is_encoder(transformer_name)
            ]
            if encoders:
                (
                    feature_encoding_steps,
                    encoded_feature_name,
                ) = self.prepare_encoding_steps(feature_name, encoders)
                encoding_steps.extend(feature_encoding_steps)
                feature_name = encoded_feature_name

            # Check for scalers
            if any(filter(is_scaler, transformer_names)):
                # WARNING: We assume scaling is the last step, maybe this is bold...
                scaler_inputs.setdefault(transformer_names[-1], []).append(feature_name)
            else:
                # If there is no scaling step, we directly add the encoded feature to
                # the model input features list.
                model_features.append(feature_name)

        for scaler_name, input_cols in scaler_inputs.items():
            scaling_steps.extend(
                (
                    sf_datalake.transform.MissingValuesDropper(
                        inputCols=input_cols,
                    ),
                    VectorAssembler(
                        inputCols=input_cols, outputCol=f"{scaler_name}_input"
                    ),
                    self.preprocessing.get_scaler(scaler_name),
                )
            )
            # Scaled features are added as a single column corresponding to the
            # associated scaler. Their names can be retrieved through dataframe schema.
            model_features.append(f"{scaler_name}_output")

        grouping_step = [
            # Filter out features that have already been assembled.
            sf_datalake.transform.MissingValuesDropper(
                inputCols=model_features,
            ),
            VectorAssembler(
                inputCols=model_features, outputCol=self.learning.features_column
            ),
        ]

        return encoding_steps + scaling_steps + grouping_step

    def prepare_encoding_steps(
        self, feature: str, encoders: List[Transformer]
    ) -> Tuple[List[Transformer], str]:
        """Generate a list of transfomer for single feature encoding.

        This helper function will set parameters for successive encoders that will be
        applied to a given feature.

        Args:
            feature: Name of the feature to be encoded.
            encoders: Sorted list of encoders.

        Returns:
            Tuple containing:
            - A list of successive encoders.
            - The name of the encoded feature.

        Raises:
            ValueError if one of the input encoders is of unknown type.

        """
        stages: List[Transformer] = []
        output_col: str = feature

        for encoder in encoders:
            encoder.setParams(inputCol=output_col)
            if isinstance(encoder, BinsOrdinalEncoder):
                suffix = "bin"
                encoder.setParams(
                    bins=self.preprocessing.ordinal_encoding_bins[feature],
                )
            elif isinstance(encoder, StringIndexer):
                suffix = "ix"
            elif isinstance(encoder, OneHotEncoder):
                suffix = "onehot"
            else:
                raise ValueError(f"Unknown type for encoder object: {encoder}.")

            output_col += f"_{suffix}"
            encoder.setParams(outputCol=output_col)
            stages.append(encoder)
        return stages, output_col
