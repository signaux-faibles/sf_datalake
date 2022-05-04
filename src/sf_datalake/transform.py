"""Utilities and classes for handling and transforming datasets."""

import logging
from itertools import chain
from typing import Iterable, List

import numpy as np
import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.sql.types import FloatType, StringType


def parse_date(
    df: pyspark.sql.DataFrame, colnames: Iterable[str]
) -> pyspark.sql.DataFrame:
    """Parses multiple columns of a pyspark.sql.DataFrame as date.

    Args:
        df: A DataFrame with dates represented as "yyyyMMdd" strings or integers.
        colnames : Names of the columns to parse.

    Returns:
        A new DataFrame with date columns as pyspark date types.

    """
    for name in colnames:
        df = df.withColumn(name, F.to_date(F.col(name).cast(StringType()), "yyyyMMdd"))
    return df


def stringify_and_pad_siren(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Normalizes the input DataFrame "siren" entries.

    Args:
        df: A DataFrame with a "siren" column, whose type can be cast to string.

    Returns:
        A DataFrame with zeros-left-padded SIREN data, as string type.

    """
    assert "siren" in df.columns, "Input DataFrame doesn't have a 'siren' column."
    df = df.withColumn("siren", df["siren"].cast("string"))
    df = df.withColumn("siren", F.lpad(df["siren"], 9, "0"))
    return df


def get_transformer(name: str) -> Transformer:
    """Gets a pre-configured Transformer object by specifying its name.

    Args:
        name: Transformer object's name

    Returns:
        The selected Transformer with prepared parameters.

    """
    factory = {
        "StandardScaler": StandardScaler(
            withMean=True,
            withStd=True,
            inputCol="to_StandardScaler",
            outputCol="from_StandardScaler",
        ),
        "OneHotEncoder": OneHotEncoder(dropLast=False),
    }
    return factory[name]


def generate_transforming_stages(config: dict) -> List[Transformer]:
    """Generates all stages related to feature transformation.

    The stages are ready to be included in a pyspark.ml.Pipeline.

    Args:
        config: model configuration, as loaded by io.load_parameters().

    Returns:
        List of prepared Transformers.

    """
    stages: List[Transformer] = []
    concat_input_cols: List[str] = []

    for transformer_name, features in config["TRANSFORMER_FEATURES"].items():
        # OneHotEncoder takes an un-assembled numeric column as input.
        if transformer_name == "OneHotEncoder":
            for feature in features:
                ohe = get_transformer(transformer_name)
                ohe.setInputCol(feature)
                ohe.setOutputCol(f"ohe_{feature}")
                concat_input_cols.append(f"ohe_{feature}")
                stages.append(ohe)
        else:
            stages.extend(
                [
                    VectorAssembler(
                        inputCols=features, outputCol=f"to_{transformer_name}"
                    ),
                    get_transformer(transformer_name),
                ]
            )
            concat_input_cols.append(f"from_{transformer_name}")

    # We add a final concatenation stage of all columns that should be fed to the model.
    stages.append(VectorAssembler(inputCols=concat_input_cols, outputCol="features"))
    return stages


def generate_preprocessing_stages(config: dict) -> List[pyspark.ml.Transformer]:
    """Generates preprocessing stages.

    These stages are ready to be included in a pyspark.ml.Pipeline object for data
    preprocessing and feature engineering.

    Args:
        config: model configuration, as loaded by io.load_parameters().

    Returns:
        A list of the preprocessing stages.

    """
    stages = [
        MissingValuesHandler(config),
        DatasetFilter(),
        PaydexColumnsAdder(config),
        SirenAggregator(config),
        AvgDeltaDebtPerSizeColumnAdder(config),
        DebtRatioColumnAdder(config),
        TargetVariableColumnAdder(),
        Covid19Adapter(config),
        DatasetColumnSelector(config),
    ]
    return stages


class AvgDeltaDebtPerSizeColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute the average change in social debt / nb of employees."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the average change in social debt / nb of employees.

        Args:
            dataset: DataFrame to transform. It should contain debt and company size
              data.

        Returns:
            Transformed DataFrame with an extra `avg_delta_dette_par_effectif` column.

        """

        mandatory_cols = {
            "montant_part_ouvriere",
            "montant_part_patronale",
            "montant_part_ouvriere_past_3",
            "montant_part_patronale_past_3",
            "effectif",
        }
        assert mandatory_cols <= set(dataset.columns)

        dataset = dataset.withColumn(
            "dette_par_effectif",
            (dataset["montant_part_ouvriere"] + dataset["montant_part_patronale"])
            / dataset["effectif"],
        )
        dataset = dataset.withColumn(
            "dette_par_effectif_past_3",
            (
                dataset["montant_part_ouvriere_past_3"]
                + dataset["montant_part_patronale_past_3"]
            )
            / dataset["effectif"],
        )
        dataset = dataset.withColumn(
            "avg_delta_dette_par_effectif",
            (dataset["dette_par_effectif"] - dataset["dette_par_effectif_past_3"]) / 3,
        )
        drop_columns = ["dette_par_effectif", "dette_par_effectif_past_3"]

        if self.config["FILL_MISSING_VALUES"]:
            dataset = dataset.fillna(
                {
                    "avg_delta_dette_par_effectif": self.config["DEFAULT_VALUES"][
                        "avg_delta_dette_par_effectif"
                    ]
                }
            )
        else:
            dataset = dataset.dropna(subset=["avg_delta_dette_par_effectif"])

        return dataset.drop(*drop_columns)


class DebtRatioColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute the debt ratio."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the debt ratio.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with an extra `ratio_dette` column.

        """

        assert {
            "montant_part_ouvriere",
            "montant_part_patronale",
            "cotisation_moy12m",
        } <= set(dataset.columns)

        dataset = dataset.withColumn(
            "ratio_dette",
            (dataset.montant_part_ouvriere + dataset.montant_part_patronale)
            / dataset.cotisation_moy12m,
        )
        if self.config["FILL_MISSING_VALUES"]:
            dataset = dataset.fillna(
                {"ratio_dette": self.config["DEFAULT_VALUES"]["ratio_dette"]}
            )
        else:
            dataset = dataset.dropna(subset=["ratio_dette"])

        return dataset


class PaydexColumnsAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute features associated with Paydex data."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the yearly variation and quantile bin of payment delay (in days).

        DataFrame to transform containing "paydex_nb_jours" and
        "paydex_nb_jours_past_12" columns.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with extra `paydex_yoy` and `paydex_bins` columns.

        """
        if not ({"paydex_bin", "paydex_yoy"} & set(self.config["FEATURES"])):
            logging.info(
                "Paydex data was not requested as a feature inside the provided \
                configuration file."
            )
            return dataset

        assert {"paydex_nb_jours", "paydex_nb_jours_past_12"} <= set(dataset.columns)

        ## Paydex variation
        dataset = dataset.withColumn(
            "paydex_yoy",
            dataset["paydex_nb_jours"] - dataset["paydex_nb_jours_past_12"],
        )

        if not "paydex_bin" in self.config["FEATURES"]:
            return dataset

        ## Binned paydex delay
        days_bins = self.config["ONE_HOT_CATEGORIES"]["paydex_bin"]
        days_splits = np.unique(np.array([float(v) for v in chain(*days_bins)]))

        bucketizer = pyspark.ml.feature.Bucketizer(
            splits=days_splits,
            handleInvalid="error",
            inputCol="paydex_nb_jours",
            outputCol="paydex_bin",
        )
        dataset = bucketizer.transform(dataset)

        ## Add corresponding 'meso' column names to the configuration.
        self.config["MESO_GROUPS"]["paydex_bin"] = [
            f"paydex_bin_ohcat{i}" for i, _ in enumerate(days_bins)
        ]

        ## Fill missing values
        if self.config["FILL_MISSING_VALUES"]:
            dataset = dataset.fillna(
                {
                    "paydex_bin": self.config["DEFAULT_VALUES"]["paydex_bin"],
                    "paydex_yoy": self.config["DEFAULT_VALUES"]["paydex_yoy"],
                }
            )
        else:
            dataset = dataset.dropna(subset=["paydex_bin", "paydex_yoy"])
        return dataset


class MissingValuesHandler(Transformer):  # pylint: disable=R0903
    """A transformer to handle missing values."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Fills or drop entries containing missing values.

        If `FILL_MISSING_VALUES` config field is true, missing data is filled with
        predefined values from the `DEFAULT_VALUES` config field.

        Args:
            dataset: DataFrame to transform containing missing values.

        Returns:
            DataFrame where previously missing values are filled, or corresponding
              entries are dropped.

        """
        assert {"FEATURES", "FILL_MISSING_VALUES", "DEFAULT_VALUES"} <= set(self.config)
        assert "time_til_failure" in dataset.columns

        if self.config["FILL_MISSING_VALUES"]:
            dataset = dataset.fillna(
                {
                    k: v
                    for (k, v) in self.config["DEFAULT_VALUES"].items()
                    if k in dataset.columns
                }
            )
        else:
            dataset = dataset.fillna(
                {
                    "time_til_failure": self.config["DEFAULT_VALUES"][
                        "time_til_failure"
                    ],
                }
            )
            dataset = dataset.dropna(
                subset=[x for x in self.config["FEATURES"] if x in dataset.columns]
            )

        return dataset


class SirenAggregator(Transformer):  # pylint: disable=R0903
    """A transformer to aggregate data at a SIREN level."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Aggregate data at a SIREN level by sum or average.

        Args:
            dataset: DataFrame to transform containing data at a SIRET level.

        Returns:
            Transformed DataFrame at a SIREN level.

        """
        assert {"IDENTIFIERS", "FEATURES", "VARIABLE_AGGREGATION"} <= set(self.config)
        assert {"siren", "periode"} <= set(dataset.columns)

        siren_lvl_colnames = [
            feat
            for feat in self.config["FEATURES"]
            if (feat not in self.config["VARIABLE_AGGREGATION"])
            and (feat in dataset.columns)
        ]

        # check if siren level features have a unique value by (siren, periode)
        n_duplicates = (
            dataset.select(["siren", "periode"] + siren_lvl_colnames)
            .dropDuplicates()
            .groupBy(["siren", "periode"])
            .count()
            .filter("count > 1")
            .count()
        )
        assert n_duplicates == 0, (
            "One or more siren level features have multiple values by "
            f"(siren, periode). siren level features: {siren_lvl_colnames}"
        )

        gb_colnames = self.config["IDENTIFIERS"] + siren_lvl_colnames

        dataset = dataset.groupBy(gb_colnames).agg(self.config["VARIABLE_AGGREGATION"])
        for colname, func in self.config["VARIABLE_AGGREGATION"].items():
            dataset = dataset.withColumnRenamed(f"{func}({colname})", colname)

        return dataset


class TargetVariableColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to aggregate data at a SIREN level."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Create the objective variable `failure_within_18m` and cast it as integer.

        Args:
            dataset: DataFrame to transform containing `time_til_failure` variable.

        Returns:
            Transformed DataFrame with an extra `failure_within_18m` column.

        """
        assert "time_til_failure" in dataset.columns

        dataset = dataset.withColumn(
            "failure_within_18m", (dataset["time_til_failure"] <= 18).cast("integer")
        )  # Models except integer or floating labels.
        return dataset


class DatasetColumnSelector(Transformer):  # pylint: disable=R0903
    """A transformer to select the columns of the dataset used in the model."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Select the columns of the dataset used in the model.

        Args:
            dataset: DataFrame to select columns from.

        Returns:
            Transformed DataFrame.

        """
        assert {"IDENTIFIERS", "FEATURES", "TARGET"} <= set(self.config)

        dataset = dataset.select(
            *(
                set(
                    self.config["IDENTIFIERS"]
                    + list(self.config["FEATURES"])
                    + self.config["TARGET"]
                )
            )
        )
        return dataset


class DatasetFilter(Transformer):  # pylint: disable=R0903
    """A transformer to filter the dataset."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Filters out small companies or public institution from a dataset.

        Only keeps private companies with more than 10 employees.

        Args:
            dataset: DataFrame to transform/filter.

        Returns:
            Transformed/filtered DataFrame.

        """
        assert {"effectif", "code_naf"} <= set(dataset.columns)
        dataset_for_filtering = (
            dataset.filter("code_naf NOT IN ('O', 'P')")
            .select(["siren", "periode", "effectif"])
            .groupBy(["siren", "periode"])
            .agg({"effectif": "sum"})
            .filter("sum(effectif) >= 10")
            .drop("sum(effectif)")
        )

        df = dataset.join(
            dataset_for_filtering, on=["siren", "periode"], how="inner"
        ).filter("code_naf NOT IN ('O', 'P')")

        return df


class ProbabilityFormatter(Transformer):  # pylint: disable=R0903
    """A transformer to format the probability column in output of a model."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Extract the positive probability and cast it as float.

        Args:
            dataset: DataFrame to transform

        Returns:
            Transformed DataFrame with casted probability data.

        """
        transform_udf = F.udf(lambda v: float(v[1]), FloatType())
        return dataset.withColumn("probability", transform_udf("probability"))


class Covid19Adapter(Transformer):  # pylint: disable=R0903
    """Adapt post-pandemic data using linear fits of features quantiles."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Adapts post-pandemic data using linear fits of features quantiles.

        Adapt post-pandemic event data through linear fits of the pre-pandemic quantiles
        --> post-pandemic quantiles mappings (for each variable that ).

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with post-pandemic data adapted to fit pre-pandemic
              distribution.

        """
        if not self.config["USE_COVID19ADAPTER"]:
            return dataset

        assert set(self.config["FEATURES_TO_ADAPT"]) <= set(dataset.columns)
        for feat in self.config["FEATURES_TO_ADAPT"]:
            dataset = dataset.withColumn(
                feat,
                F.when(
                    F.col("periode") > self.config["PANDEMIC_EVENT_DATE"],
                    self.config["COVID_ADAPTER_PARAMETERS"][feat]["params"][0]
                    + self.config["COVID_ADAPTER_PARAMETERS"][feat]["params"][1]
                    * F.col(feat),
                ).otherwise(F.col(feat)),
            )
        return dataset
