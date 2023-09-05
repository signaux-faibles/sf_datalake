"""Configuration helper classes."""

import datetime as dt
import inspect
import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from os import path
from typing import Any, Dict, Iterable, List, Tuple

import importlib_metadata
import importlib_resources
import pyspark.sql

import sf_datalake.utils


@dataclass
class LearningConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    target: Dict[str, Any] = field(
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
    model: Dict[str, Any] = None
    train_test_split_ratio: float = 0.8


@dataclass
class PreprocessingConfiguration:
    """# TODO: FILL THIS DOCSTRING"""

    identifiers: List[str] = field(default_factory=lambda: ["siren", "periode"])
    features_transformers: Dict[str, List[str]] = None
    time_aggregation: Dict[str, Dict[str, List[int]]] = None
    siren_aggregation: Dict[str, str] = field(
        default_factory=lambda: {
            "cotisation": "sum",
            "montant_part_ouvriere": "sum",
            "montant_part_patronale": "sum",
            "effectif": "sum",
            "apart_heures_consommees": "sum",
        }
    )
    # Missing values handling
    fill_missing_values: bool = True
    fill_default_values: Dict[str, Any] = None
    fill_imputation_strategy: Dict[str, Any] = None


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
    dataset_path: str = field(init=False)
    output_directory: str = field(init=False)
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

        # Check that everything is ready
        ## check()

    def override(self, source: Dict[str, Any]):
        """Override config attributes using external source.

        Args:
            source: Mapping that holds configuration data.

        """
        for obj in inspect.getmembers(self, predicate=inspect.isclass):
            for attr_name, attr_value in source.items():
                if hasattr(obj, attr_name):
                    setattr(obj, attr_name, attr_value)

    def dump(self, dump_keys: Iterable[str] = None):
        """Dumps a subset of the configuration used during a prediction run.

        Args:
            dump_keys: An Iterable of configuration parameters that should be dumped.
              All elements of `dump_keys` must be attributes of this
              ConfigurationHelper object.

        """
        spark = sf_datalake.utils.get_spark_session()

        complete_dump: Dict[str, Any] = dict(
            (attr, asdict(value))
            for attr, value in inspect.getmembers(self, predicate=is_dataclass)
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
