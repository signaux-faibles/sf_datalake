"""Utilities and classes for handling and transforming datasets."""

import datetime as dt
import itertools
from typing import Iterable, List

import numpy as np
import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml.param.shared import HasInputCol, HasInputCols, Param, Params
from pyspark.sql import Window
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


class DeltaDebtPerWorkforceColumnAdder(Transformer):
    """A transformer to compute the change in social debt / nb of employees.

    The change is normalized by the chosen duration (in months).

    Args:
        n_months: Number of months over which the diff is computed. Defaults to 3.

    """

    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_months",
        "Number of months for moving average computation.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(n_months=3)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this DeltaDebtPerWorkforceColumnAdder transformer.

        Args:
            n_months (int or list): Number of months that will be considered for diff.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Computes the average change in social debt / nb of employees.

        Args:
            dataset: DataFrame to transform. It should contain debt and workforce data.

        Returns:
            Transformed DataFrame with an extra `delta_dette_par_effectif` column.

        """
        assert {
            "montant_part_ouvriere",
            "montant_part_patronale",
            "effectif",
        } <= set(dataset.columns)

        n_months = self.getOrDefault("n_months")
        dataset = dataset.withColumn(
            "dette_par_effectif",
            (dataset["montant_part_ouvriere"] + dataset["montant_part_patronale"])
            / dataset["effectif"],
        )
        dataset = (
            DiffOperator(
                inputCol="dette_par_effectif",
                n_months=n_months,
                normalize=True,
            )
            .transform(dataset)
            .withColumnRenamed(
                "dette_par_effectif_diff{n_months}m", "delta_debt_par_effectif"
            )
        )

        return dataset


class DebtRatioColumnAdder(Transformer):
    """A transformer to compute the social debt/contribution ratio."""

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Computes the social debt/contribution ratio.

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

        return dataset.withColumn(
            "ratio_dette",
            (dataset["montant_part_ouvriere"] + dataset["montant_part_patronale"])
            / dataset["cotisation_moy12m"],
        )


class PaydexOneHotEncoder(Transformer):
    """A transformer to compute one-hot encoded features associated with Paydex data."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Computes the quantile bins of payment delay (in days).

        Args:
            dataset: DataFrame to transform. It should contain "paydex_nb_jours" and
              "paydex_nb_jours_lag12" columns.

        Returns:
             Transformed DataFrame with extra `paydex_bins` columns.

        """

        assert {"paydex_nb_jours", "paydex_nb_jours_lag12m"} <= set(dataset.columns)

        ## Binned paydex
        if "paydex_bin" in self.config["FEATURES"]:
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


class TimeNormalizer(Transformer, HasInputCols):  # pylint: disable=R0903
    """A transformer that normalizes data using corresponding time-spans.

    The duration associated with a data will be expressed through the `start` and `end`
    column names parameters. Columns that should undergo normalization are set using the
    `inputCols` parameters and will be overwritten by the transformation.

    Args:
        inputCols: A list of the columns that will be normalized.
        start: The columns that holds start dates of periods.
        end: The columns that holds end dates of periods.

    """

    start = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "start",
        "Column holding start dates",
    )
    end = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "end",
        "Column holding end dates",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCols=None, start=None, end=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this TimeNormalizer.

        Args:
            inputCols: A list of the columns that will be normalized.
            start: The columns that holds start dates of periods.
            end: The columns that holds end dates of periods.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Normalize data using associated time-spans.

        Returns:
            DataFrame with some time-normalized columns.

        """
        for param in ["inputCols", "start", "end"]:
            if self.getOrDefault(param) is None:
                raise ValueError(f"Parameter {param} is not set.")
        for col in self.getInputCols():
            dataset = dataset.withColumn(
                col,
                F.col(col)
                / F.datediff(
                    F.col(self.getOrDefault("end")), F.col(self.getOrDefault("start"))
                ),
            )
        return dataset


class MovingAverage(Transformer, HasInputCol):
    """A transformer that computes moving averages of time-series variables.

    Args:
        inputCol: The column that will be averaged.
        n_months: Number of months over which the average is computed.

    """

    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_months",
        "Number of months for moving average computation.",
    )
    ref_date = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "ref_date",
        "A reference date, used to compute number of months between rows.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol=None, n_months=None, ref_date=dt.date(2014, 1, 1))
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this MovingAverage transformer.

        Args:
            inputCol (str): The column that will be averaged.
            n_months (int or list): Number of months over which the average is computed.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute moving averages and add the corresponding new columns.

        The variable for which moving averages should be computed is expected to be
        defined through the `inputCol` parameter. The number of months over which the
        average is computed is defined through `n_months`. If `n_months` is a list, then
        for each list element, a moving average over the associated number of months
        will be produced.

        Args:
            dataset: DataFrame to transform containing time-series data.

        Returns:
            DataFrame with new "var_moy[n]m" columns, where [n] is a number of months
            over which `var`'s average is computed.

        """
        n_months = self.getOrDefault("n_months")
        if isinstance(n_months, int):
            n_months = [n_months]
        elif isinstance(n_months, list):
            pass
        else:
            raise ValueError("`n_months` should either be an int or a list of ints.")

        dataset = dataset.withColumn(
            "ref_date", F.lit(self.getOrDefault("ref_date"))
        ).withColumn(
            "months_from_ref", F.months_between("periode", "ref_date").cast("int")
        )

        time_windows = {
            n: Window()
            .partitionBy("siren")
            .orderBy(F.col("months_from_ref").asc())
            .rangeBetween(-n, Window.currentRow)
            for n in n_months
        }
        feat = self.getOrDefault("inputCol")
        for n in n_months:
            dataset = dataset.withColumn(
                f"{feat}_moy{n}m",
                F.avg(F.col(feat)).over(time_windows[n]),
            )

        return dataset.drop("ref_date", "months_from_ref")


class LagOperator(Transformer, HasInputCol):
    """A transformer that computes lagged values of a given time-indexed variable.

    Args:
        inputCol: The column that will be used to derive lagged variables.
        n_months: Number of months that will be considered for lags.

    """

    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_months",
        "Number of months for lag computation.",
    )

    ref_date = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "ref_date",
        "A reference date, used to compute number of months between rows.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol=None, n_months=None, ref_date=dt.date(2014, 1, 1))
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this LagOperator transformer.

        Args:
            inputCol (str): The column that will be used to derive lagged variables.
            n_months (int or list): Number of months that will be considered for lags.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute lagged values and add the corresponding new columns.

        The variable for which lagged values should be computed is expected to be
        defined through the `inputCol` parameter. The number of months over which the
        lag is computed is defined through `n_months`. If `n_months` is a list, then
        for each list element, an `n_months` lagged version of the `inputCol` will be
        produced.

        Args:
            dataset: DataFrame to transform containing time-series data.

        Returns:
            DataFrame with new "var_lag[n]m" columns, where [n] is a number of months
            over which `var`'s lag is computed.

        """
        input_col = self.getOrDefault("inputCol")
        n_months = self.getOrDefault("n_months")
        if isinstance(n_months, int):
            n_months = [n_months]
        elif isinstance(n_months, list):
            pass
        else:
            raise ValueError("`n_months` should either be an int or a list of ints.")

        dataset = dataset.withColumn(
            "ref_date", F.lit(self.getOrDefault("ref_date"))
        ).withColumn(
            "months_from_ref", F.months_between("periode", "ref_date").cast("int")
        )

        lag_window = (
            Window().partitionBy("siren").orderBy(F.col("months_from_ref").asc())
        )
        for n in n_months:
            dataset = dataset.withColumn(
                f"{input_col}_lag{n}m",
                F.lag(F.col(input_col), n, 0.0).over(lag_window),
            )

        return dataset.drop("ref_date", "months_from_ref")


class DiffOperator(Transformer, HasInputCol):
    """A transformer that computes the time evolution of a given time-indexed variable.

    This transformer creates a LagOperator under the hood if the required lagged
    variable is not found in the dataset.

    Args:
        inputCol: The column that will be used to derive the diff.
        n_months: Number of months that will be considered for the difference.
        normalize: If True, normalize computed difference by its duration.

    """

    normalize = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "normalize",
        "Normalize difference by its duration.",
    )
    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_months",
        "Number of months for lag computation.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol=None, n_months=None, normalize=False)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this LagOperator transformer.

        Args:
            inputCol (str): The column that will be used to derive lagged variables.
            n_months (int or list): Number of months that will be considered for lags.
            normalize (bool): If True, normalize computed difference by its duration.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute time difference(s) and add the corresponding new columns.

        The variable for which time differences should be computed is expected to be
        defined through the `inputCol` parameter. The number of months over which the
        diff is computed is defined through `n_months`. If `n_months` is a list, then
        for each list element, an `n_months` difference of the `inputCol` will be
        produced.

        Args:
            dataset: DataFrame to transform containing time-series data.

        Returns:
            DataFrame with new "var_diff[n]m" columns, where [n] is a number of months
            over which `var`'s diff is computed.

        """
        input_col = self.getOrDefault("inputCol")
        n_months = self.getOrDefault("n_months")
        if isinstance(n_months, int):
            n_months = [n_months]
        elif isinstance(n_months, list):
            pass
        else:
            raise ValueError("`n_months` should either be an int or a list of ints.")
        norm_coeff = [1 / n if self.getOrDefault("normalize") else 1 for n in n_months]

        # Compute lagged variables if needed
        missing_lags = [
            n for n in n_months if f"{input_col}_lag{n}m" not in dataset.columns
        ]
        dataset = PipelineModel(
            [LagOperator(inputCol=input_col, n_months=n) for n in missing_lags]
        ).transform(dataset)

        # Compute diffs
        for n in n_months:
            dataset = dataset.withColumn(
                f"{input_col}_diff{n}m",
                (F.col(f"{input_col}") - F.col(f"{input_col}_lag{n}m")) * norm_coeff[n],
            )

        return dataset.drop(*[f"{input_col}_lag{n}m" for n in missing_lags])


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

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Filters out samples that do not have paydex data.

        Args:
            dataset: DataFrame to filter.

        Returns:
            Filtered DataFrame.

        """
        return dataset.filter(
            F.col("paydex_nb_jours").isNotNull()
            & F.col("paydex_nb_jours_lag12").isNotNull()
        )


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
