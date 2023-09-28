# pylint: disable=unsubscriptable-object
"""Configuration helper classes."""

import dataclasses
import datetime as dt
import inspect
import json
import random
from dataclasses import dataclass
from os import path
from typing import Any, Dict, Iterable, List, Tuple

import importlib_metadata
import importlib_resources
import pyspark.sql
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


@dataclass
class LearningConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    target: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "class_col": "failure",
            "n_months": 18,
            "judgment_date_col": "date_jugement",
            "oversampling_ratio": 0.2,
        }
    )
    train_dates: Tuple[str] = ("2016-01-01", "2019-05-31")
    test_dates: Tuple[str] = ("2019-06-01", "2020-01-31")
    prediction_date: str = "2020-02-01"
    train_test_split_ratio: float = 0.8
    model_name: str = "LogisticRegression"
    model_params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "LogisticRegression": {
                "regParam": 0.12,
                "maxIter": 50,
                "tol": 1e-05,
                "standardization": False,
            },
        }
    )
    feature_column: str = "features"


@dataclass
class PreprocessingConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    identifiers: List[str] = dataclasses.field(
        default_factory=lambda: ["siren", "periode"]
    )
    siren_aggregation: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "cotisation": "sum",
            "montant_part_ouvriere": "sum",
            "montant_part_patronale": "sum",
            "effectif": "sum",
            "apart_heures_consommees": "sum",
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
    encoders_params: Dict[str, Transformer] = dataclasses.field(
        default_factory=lambda: {
            "OneHotEncoder": OneHotEncoder(dropLast=False),
            "StringIndexer": StringIndexer(),
            "BinsOrdinalEncoder": BinsOrdinalEncoder(),
        }
    )
    ordinal_encoding_bins: Dict[str, List[str]] = None
    scalers_params: Dict[str, Transformer] = dataclasses.field(
        default_factory=lambda: {
            "StandardScaler": StandardScaler(
                withMean=True,
                withStd=True,
                inputCol="StandardScaler_input",
                outputCol="StandardScaler_output",
            ),
        }
    )


@dataclass
class ExplanationConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    n_train_sample: int = 5000
    n_concerning_micro: int = 3
    feature_groups: Dict[str, List[str]] = None
    meso_groups: Dict[str, List[str]] = None


@dataclass
class IOConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    root_directory: str = "/projets/TSF"
    dataset_path: str = dataclasses.field(init=False)
    output_directory: str = dataclasses.field(init=False)
    sample_ratio: float = 1.0
    random_seed: int = random.randint(0, 10000)

    def __post_init__(self):
        """ """
        self.output_directory: str = path.join(
            self.root_directory, "predictions", str(int(dt.datetime.now().timestamp()))
        )
        self.dataset_path: str = path.join(
            self.root_directory, "data/preprocessed/datasets/full_dataset"
        )


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

    # pylint: disable=not-an-iterable
    def __init__(self, config_file: str = None, cli_args: Dict[str, Any] = None):
        # Instantiate every attribute using dataclasses default values.
        self.learning = LearningConfiguration()
        self.preprocessing = PreprocessingConfiguration()
        self.explanation = ExplanationConfiguration()
        self.io = IOConfiguration()
        self.version = importlib_metadata.version("sf_datalake")

        # Parse config file
        if config_file is not None:
            with importlib_resources.files("sf_datalake.config").joinpath(
                f"{config_file}"
            ) as f:
                config_dict = json.loads(f.read_text())
            self.override(config_dict)
        # Parse CLI
        if cli_args is not None:
            self.override(cli_args)

        # Duplicate config for time-aggregated variables.
        def add_time_aggregate_features(attribute: dict):
            for operation in self.preprocessing.time_aggregation:
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
        add_time_aggregate_features(self.preprocessing.fill_default_values)
        add_time_aggregate_features(self.preprocessing.fill_imputation_strategy)

    def override(self, source: Dict[str, Any]):
        """Override configuration attributes using external source.

        We loop over all fields found inside `source`, and try to fill attributes
        of this ConfigurationHelper's attributes.

        Args:
            source: Mapping that holds configuration data.

        Raises:
            ValueError if a field that cannot be related to any (sub-)attribute is found
              in source.

        """
        for attr_name, attr_value in source.items():
            attr_is_set = False
            for _, dcls in inspect.getmembers(self, predicate=dataclasses.is_dataclass):
                if hasattr(dcls, attr_name):
                    setattr(dcls, attr_name, attr_value)
                    attr_is_set = True
                    break
            if not attr_is_set:
                raise ValueError(
                    f"Attribute '{attr_name}' could not be matched against any "
                    "ConfigurationHelper attribute."
                )

    def get_model(self) -> Estimator:
        """Returns an Estimator object ready to be used for a learning procedure.

        Returns:
            The selected Model instantiated using config parameters.

        """
        model_factory = {
            "LogisticRegression": LogisticRegression,
            "GBTClassifier": GBTClassifier,
            "RandomForestClassifier": RandomForestClassifier,
        }
        # pylint: disable=not-a-mapping
        return (
            model_factory[self.learning.model_name]
            .setParams(**self.learning.model_params)
            .setFeaturesCol(self.learning.feature_column)
            .setLabelCol(self.learning.target["class_col"])
        )

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

        dump_dict = (
            {k: complete_dump[k] for k in dump_keys}
            if dump_keys is not None
            else complete_dump
        )
        config_df = spark.createDataFrame(pyspark.sql.Row(dump_dict))
        config_df.repartition(1).write.json(
            path.join(self.io.output_directory, "run_configuration.json")
        )

    def transforming_stages(self) -> List[Transformer]:
        """Generates all stages related to feature transformation.

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

        for (
            feature_name,
            transformer_names,
        ) in self.preprocessing.features_transformers.items():
            # Encoding
            encoders = [
                self.preprocessing.encoders_params[transformer_name]
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
                model_features.append(feature_name)

        for scaler_name, input_cols in scaler_inputs.items():
            scaling_steps.extend(
                (
                    VectorAssembler(
                        inputCols=input_cols, outputCol=f"{scaler_name}_input"
                    ),
                    self.preprocessing.scalers_params[scaler_name],
                )
            )
            model_features.append(f"{scaler_name}_output")

        grouping_step = [
            VectorAssembler(
                inputCols=model_features, outputCol=self.learning.feature_column
            )
        ]

        return encoding_steps + scaling_steps + grouping_step

    def prepare_encoding_steps(
        self, feature: str, encoders: List[Transformer]
    ) -> Tuple[List[Transformer], str]:
        """FILL DOCSTRING

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
