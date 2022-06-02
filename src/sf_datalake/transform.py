"""Utilities and classes for handling and transforming datasets."""

import itertools
import logging
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
    df = df.withColumn("siren", F.lpad(df["siren"].cast("string"), 9, "0"))
    return df


def extract_siren_from_siret(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Infer the SIREN number from a SIRET column.

    Args:
        df: A DataFrame with a "siret" column, whose type can be cast to string.

    Returns:
        A DataFrame with zeros-left-padded SIREN data, as string type.

    """
    assert "siret" in df.columns, "Input DataFrame doesn't have a 'siret' column."
    return df.withColumn(
        "siret", F.lpad(F.col("siret").cast("string"), 14, "0")
    ).withColumn("siren", F.col("siret").substr(1, 9))


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
        WorkforceFilter(),
        HasPaydexFilter(config),
        MissingValuesHandler(config),
        PaydexColumnsAdder(config),
        AvgDeltaDebtPerSizeColumnAdder(config),
        DebtRatioColumnAdder(config),
        TargetVariableColumnAdder(),
        Covid19Adapter(config),
        DatasetColumnSelector(config),
    ]
    return stages


class AvgDeltaDebtPerSizeColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute the average change in social debt / nb of employees."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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
        paydex_features = ["paydex_yoy"]

        ## Paydex variation
        dataset = dataset.withColumn(
            "paydex_yoy",
            dataset["paydex_nb_jours"] - dataset["paydex_nb_jours_past_12"],
        )

        ## Binned paydex
        if "paydex_bin" in self.config["FEATURES"]:
            paydex_features.append("paydex_bin")
            days_bins = self.config["ONE_HOT_CATEGORIES"]["paydex_bin"]
            days_splits = np.unique(
                np.array([float(v) for v in itertools.chain(*days_bins)])
            )
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
                {feat: self.config["DEFAULT_VALUES"][feat] for feat in paydex_features}
            )
        else:
            dataset = dataset.dropna(subset=paydex_features)
        return dataset


class MissingValuesHandler(Transformer):  # pylint: disable=R0903
    """A transformer to handle missing values."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Aggregate data at a SIREN level by sum or average.

        Args:
            dataset: DataFrame to transform containing data at a SIRET level.

        Returns:
            Transformed DataFrame at a SIREN level.

        """
        assert {"IDENTIFIERS", "SIREN_AGGREGATION", "NO_AGGREGATION"} <= set(
            self.config
        )

        aggregated = dataset.groupBy(self.config["IDENTIFIERS"]).agg(
            self.config["SIREN_AGGREGATION"]
        )
        for colname, func in self.config["SIREN_AGGREGATION"].items():
            aggregated = aggregated.withColumnRenamed(f"{func}({colname})", colname)
        siren_level = dataset.select(
            self.config["NO_AGGREGATION"] + self.config["IDENTIFIERS"]
        ).distinct()
        return aggregated.join(
            siren_level,
            on=["siren", "periode"],
            how="left",
        )


class TimeNormalizer(Transformer):  # pylint: disable=R0903
    """A transformer that normalizes data using corresponding time-spans."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Normalize data by dividing variables by the associated time-span.

        The duration associated with a data should be expressed in months. Columns that
        should undergo normalization will be looked for in the configuration under the
        "TIME_NORMALIZATION" key. The expected format is:

        {duration_col: [data_cols]}, where duration_col is the name of a column that
        holds duration (in number of months) associated with the variables under
        [data_cols].

        Args:
            dataset: DataFrame to transform containing raw period-related data.

        Returns:
            DataFrame with some normalized columns.

        """
        assert {"TIME_NORMALIZATION"} <= set(self.config)

        # Normalize accounting year by duration
        for duration_col, var_cols in self.config["TIME_NORMALIZATION"].items():
            for c in var_cols:
                dataset = dataset.withColumn(c, F.col(c) / F.col(duration_col))
        return dataset


class TargetVariableColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute the company failure target variable."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Create the learning target variable `failure_within_18m`.

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

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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


class PrivateCompanyFilter(Transformer):  # pylint: disable=R0903
    """A transformer that filters a dataset according to its public/private nature."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Filters out public institutions from a dataset.

        Only keeps private companies using `code_naf` variable.

        Args:
            dataset: DataFrame to filter.

        Returns:
            Filtered DataFrame.

        """
        if "code_naf" not in dataset.columns:
            raise KeyError("Dataset has no 'code_naf' column.")
        return dataset.filter("code_naf NOT IN ('O', 'P')")


class HasPaydexFilter(Transformer):  # pylint: disable=R0903
    """A transformer that filters according to paydex data availability."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Filters out samples that do not have paydex data.

        Args:
            dataset: DataFrame to filter.

        Returns:
            Filtered DataFrame.

        """
        if {"paydex_bin", "paydex_yoy"} & set(self.config["FEATURES"]):
            logging.info(
                "Paydex data features were requested through the provided \
                configuration file. The dataset will be filtered to only keep samples \
                with available paydex data."
            )
            return dataset.filter(
                F.col("paydex_nb_jours").isNotNull()
                & F.col("paydex_nb_jours_past_12").isNotNull()
            )

        return dataset


class WorkforceFilter(Transformer):  # pylint: disable=R0903
    """A transformer to filter the dataset according to workforce size."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Filters out small companies

        Only keeps companies with more than 10 employees.

        Args:
            dataset: DataFrame to filter.

        Returns:
            Filtered DataFrame.

        """
        if "effectif" not in dataset.columns:
            raise KeyError("Dataset has no 'effectif' column.")
        return dataset.filter(F.col("effectif") >= 10)


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

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
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
