"""Utilities and classes for handling and transforming datasets."""

from typing import Dict, Iterable, List

import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.window import Window


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


def process_payment(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Computes the number of payments.

    Args:
        df: A DataFrame containing payment data.

    Returns:
        A DataFrame with a new "nb_paiement" column.

    """
    df = df.withColumn("mvt_djc_int", F.unix_timestamp(F.col("mvt_djc")))
    df = df.orderBy("frp", "art_cleart", "mvt_djc").groupBy(
        ["frp", "art_cleart", "mvt_deff"]
    )
    df = (
        df.agg(F.min("mvt_djc_int"), F.sum("mvt_mcrd"))
        .select(["frp", "art_cleart", "min(mvt_djc_int)", "sum(mvt_mcrd)"])
        .withColumnRenamed("min(mvt_djc_int)", "min_mvt_djc_int")
        .withColumnRenamed("sum(mvt_mcrd)", "sum_mvt_mcrd")
        .dropDuplicates()
    )

    windowval = (
        Window.partitionBy("art_cleart")
        .orderBy(["frp", "min_mvt_djc_int"])
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    df = (
        df.filter("sum_mvt_mcrd != 0")
        .withColumn("mnt_paiement_cum", F.sum("sum_mvt_mcrd").over(windowval))
        .withColumn("nb_paiement", F.count("sum_mvt_mcrd").over(windowval))
        .dropDuplicates()
    )
    return df


def get_scaler_from_str(s: str) -> Transformer:
    """Get a Transformer from its name.

    Args:
        s: Name of the Transformer

    Returns:
        The selected Transformer with prepared parameters
    """
    factory = {
        "StandardScaler": StandardScaler(
            withMean=True,
            withStd=True,
            inputCol="features_to_transform_StandardScaler",
            outputCol="features_transformed_StandardScaler",
        )
    }
    return factory[s]


def generate_scaling_stages(config: dict) -> List[Transformer]:
    """Generates all stages related to Transformer objects.

    The stages are ready to be included in a pyspark.ml.Pipeline.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        List of prepared Transformers.

    """
    stages: List[Transformer] = []
    transformed_features: List[str] = []
    transformer_features: Dict[str, List[str]] = {}
    for feature, transformer in config["TRANSFORMERS"]:
        transformer_features.setdefault(transformer, []).append(feature)
    for transformer, features in transformer_features.items():
        outputColAssembler = f"features_to_transform_{transformer}"
        outputColScaler = f"features_transformed_{transformer}"
        transformer_vector_assembler = VectorAssembler(
            inputCols=features, outputCol=outputColAssembler
        )
        stages += [transformer_vector_assembler, get_scaler_from_str(transformer)]
        transformed_features.append(outputColScaler)

    concat_vector_assembler = VectorAssembler(
        inputCols=transformed_features, outputCol="features"
    )
    stages.append(concat_vector_assembler)
    return stages


def generate_preprocessing_stages(config: dict) -> List[pyspark.ml.Transformer]:
    """Generates stages for preprocessing pipeline construction.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A list of the preprocessing stages.

    """
    stages = [
        MissingValuesHandler(config),
        SirenAggregator(config),
        DatasetFilter(),
        AvgDeltaDebtPerSizeColumnAdder(config),
        DebtRatioColumnAdder(config),
        TargetVariableColumnAdder(),
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


class PaydexYoyColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to compute the year over year values with Paydex data."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the year over year values with Paydex data.

        Args:
            dataset: DataFrame to transform containing "paydex_nb_jours" and
                     "paydex_nb_jours_past_12" columns.

        Returns:
            Transformed DataFrame with an extra `paydex_yoy` column.

        """
        assert {"paydex_nb_jours", "paydex_nb_jours_past_12"} <= set(dataset.columns)

        return dataset.withColumn(
            "paydex_yoy",
            dataset["paydex_nb_jours"] - dataset["paydex_nb_jours_past_12"],
        )


class PaydexGroupColumnAdder(Transformer):  # pylint: disable=R0903
    """A transformer to cut paydex number of days data into quantile bins."""

    def __init__(self, num_buckets) -> None:
        super().__init__()
        self.num_buckets = num_buckets

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Cuts paydex number of days data into quantile bins.

        Args:
            dataset: DataFrame to transform containing "paydex_nb_jours".

        Returns:
            Transformed DataFrame with an extra `paydex_bins` column.

        """
        assert "paydex_nb_jours" in dataset.columns

        qds = pyspark.ml.feature.QuantileDiscretizer(
            inputCol="paydex_nb_jours",
            outputCol="paydex_bins",
            handleInvalid="error",
            numBuckets=self.num_buckets,
        )
        return qds.fit(dataset).transform(dataset)


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
        assert {"IDENTIFIERS", "FEATURES", "AGG_DICT"} <= set(self.config)
        assert {"siren", "periode"} <= set(dataset.columns)

        siren_lvl_colnames = [
            feat
            for feat in self.config["FEATURES"]
            if (feat not in self.config["AGG_DICT"]) and (feat in dataset.columns)
        ]
        gb_colnames = self.config["IDENTIFIERS"] + siren_lvl_colnames

        dataset = dataset.groupBy(gb_colnames).agg(self.config["AGG_DICT"])
        for colname, func in self.config["AGG_DICT"].items():
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
                    + self.config["FEATURES"]
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

        return dataset.filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")


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
