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


def get_transformer(name: str) -> Transformer:
    """Get a pre-configured Transformer object by specifying its name.

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
        config: model configuration, as loaded by utils.get_config().

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
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A list of the preprocessing stages.

    """
    stages = [
        MissingValuesHandler(config),
        PaydexColumnsAdder(config),
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
                "paydex data was not requested as a feature inside the provided \
                configuration file."
            )
            return dataset

        assert {"paydex_nb_jours", "paydex_nb_jours_past_12"} <= set(dataset.columns)

        ## Paydex variation
        dataset = dataset.withColumn(
            "paydex_yoy",
            dataset["paydex_nb_jours"] - dataset["paydex_nb_jours_past_12"],
        )

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


class UnbiaserCovid19(Transformer):  # pylint: disable=R0903
    """A transformer to unbias data after COVID-19 event."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Unbias data after the COVID-19 event by a linear model fit on
        quantiles before/after COVID-19.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with unbiased data after the COVID-19 event.

        """
        FEATURES_TO_BE_UNBIASED = [  # according to Q-Q plots
            "MNT_AF_BFONC_BFR",
            "MNT_AF_BFONC_TRESORERIE",
            "RTO_AF_RATIO_RENT_MBE",
            "MNT_AF_BFONC_FRNG",
            "MNT_AF_CA",
            "MNT_AF_SIG_EBE_RET",
            "RTO_AF_RENT_ECO",
            "RTO_AF_SOLIDITE_FINANCIERE",
            "RTO_INVEST_CA",
            "cotisation",
            "effectif",
        ]

        # unbiaser_covid_params is generated from
        # exploration.generate_unbiaser_covid_params()
        unbiaser_covid_params = {
            "MNT_AF_BFONC_BFR": {
                "params": [23623.665674728258, 0.56296943266575639],
                "rmse": 70571.22084858362,
                "r2": 0.9940361612169091,
            },
            "MNT_AF_BFONC_TRESORERIE": {
                "params": [22236.184117106088, 0.58412926143396038],
                "rmse": 42171.00682101867,
                "r2": 0.9944725281058036,
            },
            "RTO_AF_RATIO_RENT_MBE": {
                "params": [-0.0015722972109895586, 1.0464419929184383],
                "rmse": 0.0014390084193837763,
                "r2": 0.999422153539393,
            },
            "MNT_AF_BFONC_FRNG": {
                "params": [59213.016524039755, 0.57295572977379372],
                "rmse": 95524.35412399008,
                "r2": 0.9952842529340152,
            },
            "MNT_AF_CA": {
                "params": [960810.9153172815, 0.41353858849763436],
                "rmse": 972975.2806476083,
                "r2": 0.9817355692218368,
            },
            "MNT_AF_SIG_EBE_RET": {
                "params": [38680.8834803771, 0.51602364042763094],
                "rmse": 63670.270806772,
                "r2": 0.9778103093621975,
            },
            "RTO_AF_RENT_ECO": {
                "params": [-0.009450544412868727, 1.1473067958388148],
                "rmse": 0.005281719432831142,
                "r2": 0.9992802110584927,
            },
            "RTO_AF_SOLIDITE_FINANCIERE": {
                "params": [-0.03858807453691644, 1.0472144997536623],
                "rmse": 0.005608115232355715,
                "r2": 0.9991168735366558,
            },
            "RTO_INVEST_CA": {
                "params": [-0.004315476083762521, 0.99394772186907443],
                "rmse": 0.004566023973095764,
                "r2": 0.9986652084031244,
            },
            "cotisation": {
                "params": [100.91665931258059, 1.1356772860529647],
                "rmse": 593.7368742398097,
                "r2": 0.9996548293427363,
            },
            "effectif": {
                "params": [-0.9358249557851157, 1.0530334074660543],
                "rmse": 0.7119727250833081,
                "r2": 0.9994210784203903,
            },
        }

        assert set(FEATURES_TO_BE_UNBIASED) <= set(dataset.columns)

        for feat in FEATURES_TO_BE_UNBIASED:
            dataset = dataset.withColumn(
                feat,
                F.when(
                    F.col("periode") > "2020-02-29",
                    unbiaser_covid_params[feat]["params"][0]
                    + unbiaser_covid_params[feat]["params"][1] * F.col(feat),
                ).otherwise(F.col(feat)),
            )

        return dataset
