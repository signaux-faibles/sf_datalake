"""Preprocessor utilities and classes. """

from functools import reduce
from typing import Iterable, List

import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql.types import StringType
from pyspark.sql.window import Window


def parse_date(
    df: pyspark.sql.DataFrame, colnames: Iterable[str]
) -> pyspark.sql.DataFrame:
    """Parse multiple columns of a pyspark.sql.DataFrame as date.

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
    """Compute the number of payments.

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


class AvgDeltaDebtPerSizeColumnAdder(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to compute the average change in social debt / nb of employees."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the average change in social debt / nb of employees.

        Args:
            dataset: DataFrame to transform containing debt and company size ("effectif") data.

        Returns:
            Transformed DataFrame with an extra `avg_delta_dette_par_effectif` column.

        """
        assert "montant_part_ouvriere" in dataset.columns
        assert "montant_part_patronale" in dataset.columns
        assert "montant_part_ouvriere_past_3" in dataset.columns
        assert "montant_part_patronale_past_3" in dataset.columns
        assert "effectif" in dataset.columns

        dataset = dataset.withColumn(
            "dette_par_effectif",
            (dataset["montant_part_ouvriere"] + dataset["montant_part_patronale"])
            / dataset["effectif"],
        )
        # TODO replace([np.nan, np.inf, -np.inf], 0)

        dataset = dataset.withColumn(
            "dette_par_effectif_past_3",
            (
                dataset["montant_part_ouvriere_past_3"]
                + dataset["montant_part_patronale_past_3"]
            )
            / dataset["effectif"],
        )
        # TODO replace([np.nan, np.inf, -np.inf], 0)

        dataset = dataset.withColumn(
            "avg_delta_dette_par_effectif",
            (dataset["dette_par_effectif"] - dataset["dette_par_effectif_past_3"]) / 3,
        )

        drop_columns = ["dette_par_effectif", "dette_par_effectif_past_3"]
        return dataset.drop(*drop_columns)


class DebtRatioColumnAdder(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to compute the debt ratio."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the debt ratio.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with an extra `ratio_dette` column.

        """
        assert "montant_part_ouvriere" in dataset.columns
        assert "montant_part_patronale" in dataset.columns
        assert "cotisation_moy12m" in dataset.columns

        dataset = dataset.withColumn(
            "ratio_dette",
            (dataset.montant_part_ouvriere + dataset.montant_part_patronale)
            / dataset.cotisation_moy12m,
        )
        return dataset


class PaydexYoyColumnAdder(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to compute the year over year values with Paydex data."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Computes the year over year values with Paydex data.

        Args:
            dataset: DataFrame to transform containing "paydex_nb_jours" and
                     "paydex_nb_jours_past_12" columns.

        Returns:
            Transformed DataFrame with an extra `paydex_yoy` column.

        """
        assert "paydex_nb_jours" in dataset.columns
        assert "paydex_nb_jours_past_12" in dataset.columns

        return dataset.withColumn(
            "paydex_yoy",
            dataset["paydex_nb_jours"] - dataset["paydex_nb_jours_past_12"],
        )


class PaydexGroupColumnAdder(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
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
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to handle missing values."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Fill missing values by the median or a specific value (see config['DEFAULT_VALUES']).

        Args:
            dataset: DataFrame to transform containing missing values.

        Returns:
            Transformed DataFrame with missing values completed.

        """
        assert "FEATURES" in self.config
        assert "FILL_MISSING_VALUES" in self.config
        assert "DEFAULT_VALUES" in self.config
        assert "time_til_failure" in dataset.columns

        ratio_variables = list(
            filter(lambda x: x[:3] == "RTO", self.config["FEATURES"])
        )
        if ratio_variables:
            ratio_default_values = {}
            data_medians = reduce(
                lambda x, y: x + y, dataset.approxQuantile(ratio_variables, [0.5], 0.05)
            )
            for var, med in zip(ratio_variables, data_medians):
                ratio_default_values[var] = med

            default_data_values = dict(
                **ratio_default_values, **self.config["DEFAULT_VALUES"]
            )
        else:
            default_data_values = self.config["DEFAULT_VALUES"]

        if self.config["FILL_MISSING_VALUES"]:
            dataset = dataset.fillna(
                {k: v for (k, v) in default_data_values.items() if k in dataset.columns}
            )
        else:
            dataset = dataset.fillna(
                {
                    "time_til_failure": self.config["DEFAULT_VALUES"][
                        "time_til_failure"
                    ],
                }
            )
            variables_to_dropna = [
                x for x in self.config["FEATURES"] if x in dataset.columns
            ]
            dataset = dataset.dropna(subset=tuple(variables_to_dropna))
        return dataset


class SirenAggregator(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
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
        assert "BASE_FEATURES" in self.config
        assert "FEATURES" in self.config
        assert "AGG_DICT" in self.config
        assert "siren" in dataset.columns
        assert "periode" in dataset.columns

        no_agg_colnames = [
            feat
            for feat in self.config["FEATURES"]
            if (not feat in self.config["AGG_DICT"]) and (feat in dataset.columns)
        ]  # already at SIREN level
        groupby_colnames = self.config["BASE_FEATURES"] + no_agg_colnames
        # TODO problem here: code_naf and time_til_failure are not unique for one
        # SIREN: example for code_naf see siren 311403976, is it ok?

        dataset = dataset.groupBy(*(set(groupby_colnames))).agg(self.config["AGG_DICT"])
        for colname, func in self.config["AGG_DICT"].items():
            if func == "mean":
                func = "avg"  # groupBy mean produces variables such as avg(colname)
            dataset = dataset.withColumnRenamed(f"{func}({colname})", colname)

        ### TODO : ratio_dette_moyenne12m should be computed from the
        ### aggregated ratio_dette variable.
        # w = dataset.groupBy("siren", F.window(dataset.periode - 365
        # days, "365 days")).avg("ratio_dette")
        return dataset


class TargetVariableColumnAdder(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
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
            "failure_within_18m", dataset["time_til_failure"] <= 18
        )
        dataset = dataset.withColumn(
            "failure_within_18m", dataset.failure_within_18m.astype("integer")
        )  # Needed  for models
        return dataset


class DatasetColumnSelector(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
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
        assert "BASE_FEATURES" in self.config
        assert "FEATURES" in self.config
        assert "TARGET_FEATURE" in self.config

        dataset = dataset.select(
            *(
                set(
                    self.config["BASE_FEATURES"]
                    + self.config["FEATURES"]
                    + self.config["TARGET_FEATURE"]
                )
            )
        )
        return dataset


class DatasetFilter(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to filter the dataset."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Filter the dataset by filtering out firms on 'effectif' and 'code_naf' variables.

        Args:
            dataset: DataFrame to transform/filter.

        Returns:
            Transformed/filtered DataFrame.

        """
        assert "effectif" in dataset.columns
        assert "code_naf" in dataset.columns

        return dataset.filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")


def generate_stages(config: dict) -> List[pyspark.ml.Transformer]:
    """Generate stage related to the preprocessing to be transformed in a preprocessing pipeline.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        List of the preprocessing stages.
    """
    stages = [
        MissingValuesHandler(config),
        SirenAggregator(config),
        AvgDeltaDebtPerSizeColumnAdder(),
        DebtRatioColumnAdder(),
        TargetVariableColumnAdder(),
        DatasetColumnSelector(config),
        DatasetFilter(),
    ]
    # TODO: "siren='347546970' AND periode='2021-01-01 01:00:00'"
    # => should be kept (was not the case before)
    # TODO: indics_annuels.filter(
    #           "siren='397988239' AND periode='2016-06-01 02:00:00'"
    #       ).select(
    #           ["siret", "siren", "periode", "code_naf", "time_til_failure",
    #            "effectif", "MNT_AF_BFONC_BFR","MNT_AF_BFONC_TRESORERIE",
    #            "RTO_AF_RATIO_RENT_MBE","MNT_AF_BFONC_FRNG","MNT_AF_CA",
    #            "MNT_AF_SIG_EBE_RET","RTO_AF_RENT_ECO","RTO_AF_SOLIDITE_FINANCIERE",
    #            "RTO_INVEST_CA"]).show()
    # => duplicate SIRET, mutiple MRV values => Need also an agg function OR Error?
    # => before effectif was aggregated and this data was kept in study but should it be?
    return stages
