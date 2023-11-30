"""Utilities and classes for handling and transforming datasets."""

import datetime as dt
import itertools
import logging
from typing import List, Tuple, Union

import numpy as np
import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import keyword_only
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.feature import Imputer
from pyspark.ml.param.shared import (
    HasInputCol,
    HasInputCols,
    HasOutputCol,
    Param,
    Params,
)
from pyspark.sql import Window

from sf_datalake.utils import count_missing_values


def vector_disassembler(
    df: pyspark.sql.DataFrame,
    columns: List[str],
    assembled_col: str,
    keep: List[str] = None,
) -> pyspark.sql.DataFrame:
    """Inverse operation of `pyspark.ml.feature.VectorAssembler` operator.

    Args:.
        df: input DataFrame.
        columns: individual columns previously assembled by a `VectorAssembler`.
        assembled_col: `VectorAssembler`'s output column name.
        keep: additional columns to keep that are not part of the assembled column.

    Returns:
        A DataFrame with columns that have been "disassembled".

    """
    if keep is None:
        keep = []
    assert set(keep + [assembled_col]) <= set(df.columns)

    def udf_vector_disassembler(col):
        return F.udf(lambda v: v.toArray().tolist(), T.ArrayType(T.DoubleType()))(col)

    df = df.select(keep + [assembled_col])
    df = df.withColumn(
        assembled_col, udf_vector_disassembler(F.col(assembled_col))
    ).select(keep + [F.col(assembled_col)[i] for i in range(len(columns))])

    for i, col in enumerate(columns):
        df = df.withColumnRenamed(f"{assembled_col}[{i}]", col)
    return df


class DateParser(
    Transformer, HasInputCol, HasOutputCol
):  # pylint: disable=too-few-public-methods
    """A transformer that parses some string timestamp / date info to pyspark date type.

    The data will be parsed from the `inputCol` into the `outputCol`. Both can be set at
    instanciation time or using either `setInputCol`, `setOutputCol` or `setParams`. The
    initial string format should be specified using a datetime pattern.

    Args:
        inputCol: The column containing data to be parsed.
        outputCol: The output column to be created.
        format: The input column datetime format. Defaults to "yyyyMMdd".

    """

    format = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "format",
        "Expected datetime format inside inputCol.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(format="yyyyMMdd")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCol: The column containing data to be parsed.
            outputCol: The output column to be created.
            format: The input column datetime format. Defaults to "yyyyMMdd".

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Extract date info from `inputCol` into `outputCol`.

        Args:
            dataset: A DataFrame with dates / datetime represented as strings.

        Returns:
            A new DataFrame with the set `outputCol` holding pyspark date types.

        """
        return dataset.withColumn(
            self.getOrDefault("outputCol"),
            F.to_date(
                F.col(self.getOrDefault("inputCol")).cast(T.StringType()),
                self.getOrDefault("format"),
            ),
        )


class BinsOrdinalEncoder(
    Transformer, HasInputCol, HasOutputCol
):  # pylint: disable=too-few-public-methods
    """A transformer that bins continuous features into ordered buckets.

    The transformed feature will be encoded as a float ordinal feature (0.0, 1.0, ...)
    and can be further one-hot encoded if needed using another Transformer.

    Args:
        inputCol (str): The variable to be encoded.
        bins (list): A list of bins, with adjacent and increasing number of days values,
          e.g.: `[[0, 2], [2, 10], [10, inf]]`. All values inside bins will be cast to
          floats.
        outputCol (str): The ordinal encoded column name.

    """

    bins = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "bins",
        "Bins edges to be used to bucketize variable.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCol (str): The variable to be encoded.
            bins (list): A list of bins, with adjacent and increasing values, e.g.:
              `[[0, 2], [2, 10], [10, inf]]`. All values inside bins will be cast to
              floats.
            outputCol (str): The ordinal encoded column name.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Ordinal encode the variable using input bins.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with extra `{var}_bins` columns.

        """

        ## Binned paydex
        bins = self.getOrDefault("bins")
        splits = np.unique(np.array([float(v) for v in itertools.chain(*bins)]))
        bucketizer = pyspark.ml.feature.Bucketizer(
            splits=splits,
            handleInvalid="error",
            inputCol=self.getOrDefault("inputCol"),
            outputCol=self.getOrDefault("outputCol"),
        )
        return bucketizer.transform(dataset)


class MissingValuesHandler(
    Transformer, HasInputCols
):  # pylint: disable=too-few-public-methods
    """A transformer to handle missing values.

    Uses pyspark.sql.DataFrame.fillna, statistical Imputer object or backward/forward
    filling to fill missing values. Use either the `value` or `strategy` argument as
    both are mutually exclusive. If `strategy` is set to bfill or ffill, missing values
    are filled by using the next valid observation to fill the gap in the forward (resp.
    backward) direction.

    Median convention: for an even number of samples, the median is computed as the
    `n/2`th sample.

    Args:
        inputCols: The input dataset columns to consider for filling.
        value: Value to replace null values with. It must be a mapping from column name
          (string) to replacement value. The replacement value must be an int, float,
          boolean, or string.
        strategy: For statistical imputation, use : 'mean', 'median' or 'mode'. For
          forward / backward filling use 'ffill'/'bfill'

    """

    value = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "value",
        "Value to replace null values with.",
    )
    strategy = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "strategy",
        "Method to use for imputation.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(value=None, strategy=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCols (list[str]): The input dataset columns to consider for filling.
            value (dict): Value to replace null values with. It must be a mapping from
              column name to replacement value. If None, stat imputation is applied.
            strategy (str): For statistical imputation, use : 'mean', 'median' or
              'mode'. For forward / backward filling use 'ffill'/'bfill'
        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Fill entries containing missing values inplace.

        Args:
            dataset: DataFrame to transform containing missing values.

        Returns:
            DataFrame where previously missing values are filled using selected method.

        Raises:
            ValueError if:
            - both `value` and `strategy` or neither are set.
            - statistical imputation is required on non-numerical columns.

        """
        strategy: str = self.getOrDefault("strategy")
        value: dict = self.getOrDefault("value")
        input_cols: List[str] = self.getOrDefault("inputCols")

        # Check arguments
        if value is not None and strategy is not None:
            raise ValueError(
                "`value` and `strategy` are mutually exclusive. Use either one."
            )
        if value is None and strategy is None:
            raise ValueError("Either `value` or `strategy` must be set.")

        # Use values
        if value is not None:
            return dataset.fillna(
                {var: val for var, val in value.items() if var in input_cols}
            )

        # Use a strategy
        ## Filling strategy
        if strategy in ["bfill", "ffill"]:
            forward_window = (
                Window()
                .partitionBy("siren")
                .orderBy(F.col("periode").asc())
                .rowsBetween(Window.currentRow, Window.unboundedFollowing)
            )
            backward_window = (
                Window()
                .partitionBy("siren")
                .orderBy(F.col("periode").asc())
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            fill_window = forward_window if strategy == "bfill" else backward_window
            lookup_function = F.first if strategy == "bfill" else F.last
            for col in input_cols:
                dataset = dataset.withColumn(
                    col, lookup_function(col, ignorenulls=True).over(fill_window)
                )
        ## Statistical imputation
        if strategy in ["median", "mean", "mode"]:
            # Check that columns are of numerical type and non-empty
            if any(
                dtype in {"bool", "timestamp", "date", "string"}
                for dtype in [dtype for _, dtype in dataset.select(input_cols).dtypes]
            ):
                raise ValueError(
                    "Statistical imputation of a non-numerical variable is not "
                    "supported."
                )
            n_rows = dataset.count()
            n_missing_values = count_missing_values(dataset)
            for col in input_cols:
                if n_missing_values[col] == n_rows:
                    raise ValueError(
                        "Statistical imputation of a null column is not supported."
                    )

            imputer = Imputer(strategy=strategy)
            imputer.setInputCols(input_cols)
            imputer.setOutputCols(input_cols)
            dataset = imputer.fit(dataset).transform(dataset)
        return dataset


class MissingValuesDropper(
    Transformer, HasInputCols
):  # pylint: disable=too-few-public-methods
    """Drops missing values.

    This Transformer is only a way to simply drop null values inside a Pipeline.
    pyspark.sql.DataFrame.dropna() is called using `how=any`, meaning that any row
    containing at least one missing value found among `inputCols` will be dropped.

    Args:
        inputCols: The input dataset columns to consider for dropping.
        ignore_type: Ignore any inputCol if its type is found inside ignore_type.

    """

    ignore_type = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "ignore_type",
        "Columns of these types will be ignored.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            ignore_type=(
                T.ArrayType,
                T.MapType,
                T.StructType,
                T.StructField,
                T.UserDefinedType,
            )
        )
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCols (list[str]): The input dataset columns to consider for dropping.
            ignore_type (tuple[str]): Ignore any inputCol if its type is found inside
              ignore_type.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies dropna to a dataset.

        Args:
            dataset: DataFrame containing missing values.

        Returns:
            DataFrame where rows with missing values are dropped.

        """
        input_cols: List[str] = self.getOrDefault("inputCols")
        ignore_type: Tuple[str] = self.getOrDefault("ignore_type")
        dropna_dataset = dataset.dropna(
            subset=[
                feature
                for feature in input_cols
                if not isinstance(dataset.schema[feature].dataType, ignore_type)
            ]
        )

        if dropna_dataset.count() != dataset.count():
            logging.info(
                "Some rows containing null values in subset %s were dropped",
                input_cols,
            )
        return dropna_dataset


class IdentifierNormalizer(
    Transformer, HasInputCol
):  # pylint: disable=too-few-public-methods
    """A transformer that normalizes a DataFrame's SIREN / SIRET data.

    It does so by:
    - Casting the identifier column values to strings.
    - Left-padding the identifier with zeroes.

    The zero-padding is done inplace.

    Args:
        inputCol: The column containing identifier to normalize. Default to "siren".
        n_pad: Length of string to be zero-padded. A SIREN is 9 characters long, while
          a SIRET is 14 characters long.

    """

    n_pad = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_pad",
        "Length of string to be zero-padded.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol="siren", n_pad=9)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this IdentifierNormalizer.

        Args:
            inputCol: The column containing SIRENs to normalize. Default to "siren".
            n_pad: Length of string to be zero-padded. A SIREN is 9 characters long,
              while a SIRET is 14 characters long.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Normalize identifier data inplace.

        Args:
            dataset: A DataFrame with an identifier (e.g. SIREN data) column, whose type
              can be cast to string.

        Returns:
            A DataFrame with zeros-left-padded identifier data, as string type.

        """
        identifier = self.getOrDefault("inputCol")
        n_pad = self.getOrDefault("n_pad")
        assert (
            identifier in dataset.columns
        ), f"Input DataFrame doesn't have a {identifier} column."
        return dataset.withColumn(
            identifier, F.lpad(dataset[identifier].cast(T.StringType()), n_pad, "0")
        )


class SiretToSiren(
    Transformer, HasInputCol, HasOutputCol
):  # pylint: disable=too-few-public-methods
    """A transformer that generates a SIREN column from SIRET information.

    It does so by:
    - Casting the SIRET column values to strings.
    - Left-padding the SIRET with zeroes.
    - Extracting the first 9 digits.

    Args:
        inputCol: The column containing SIRET values. Default to "siret".
        outputCol: The column containing SIREN values. Default to "siren".

    """

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol="siret", outputCol="siren")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this SiretToSiren.

        Args:
            inputCol: The column containing SIRET values. Default to "siret".
            outputCol: The column containing SIREN values. Default to "siren".

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Extract SIREN data.

        Args:
            dataset: A DataFrame with a "siret" column, whose type can be cast to
              string.

        Returns:
            A DataFrame with zeros-left-padded SIREN data, as string type.

        """
        siret_col = self.getOrDefault("inputCol")
        siren_col = self.getOrDefault("outputCol")

        assert (
            siret_col in dataset.columns
        ), f"Input DataFrame doesn't have a {siret_col} column."
        return dataset.withColumn(
            siren_col,
            F.lpad(F.col(siret_col).cast(T.StringType()), 14, "0").substr(1, 9),
        )


class SirenAggregator(Transformer):  # pylint: disable=too-few-public-methods
    """A transformer to aggregate data at a SIREN level."""

    grouping_cols = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "grouping_cols",
        "Column to group by begore aggregation",
    )
    aggregation_map = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "aggregation_map",
        "Mapping between feature names and an aggregation operation.",
    )
    no_aggregation = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "no_aggregation",
        "Column that should be kept as-is.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            grouping_cols=["siren", "periode"],
            aggregation_map=None,
            no_aggregation=None,
        )
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this SiretToSiren.

        Args:
            grouping_cols (list[str]): A list of columns to groupby before aggregation.
            aggregation_map (dict[str, str]): A mapping between variables names and
              aggregation operation.
            no_aggregation (list[str]): A list of columns that should not be aggregated
              but should be preserved in the output.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Aggregate data at a SIREN level by sum, average, etc.

        Args:
            dataset: DataFrame to transform containing data at a SIRET level.

        Returns:
            Transformed DataFrame at a SIREN level.

        """

        grouping_cols = self.getOrDefault("grouping_cols")
        no_aggregation = self.getOrDefault("no_aggregation")
        agg_map = self.getOrDefault("aggregation_map")
        if no_aggregation is None:
            no_aggregation = []

        aggregated = dataset.groupBy(grouping_cols).agg(agg_map)
        for colname, func in agg_map.items():
            aggregated = aggregated.withColumnRenamed(f"{func}({colname})", colname)
            siren_level = dataset.select(grouping_cols + no_aggregation).distinct()
        return aggregated.join(
            siren_level,
            on=self.getOrDefault("grouping_cols"),
            how="left",
        )


class TimeNormalizer(
    Transformer, HasInputCols
):  # pylint: disable=too-few-public-methods
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


class MovingAverage(
    Transformer, HasInputCol
):  # pylint: disable=too-few-public-methods,protected-access
    """A transformer that computes moving averages of time-series variables.

    Args:
        inputCol: The column that will be averaged.
        n_months: Number of months over which the average is computed.

    """

    n_months = Param(
        Params._dummy(),
        "n_months",
        "Number of months for moving average computation.",
    )
    ref_date = Param(
        Params._dummy(),
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
            DataFrame with new "var_mean[n]m" columns, where [n] is a number of months
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
            "months_from_ref",
            F.months_between("periode", "ref_date").cast(T.IntegerType()),
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
                f"{feat}_mean{n}m",
                F.avg(F.col(feat)).over(time_windows[n]),
            )

        return dataset.drop("ref_date", "months_from_ref")


class LagOperator(
    Transformer, HasInputCol
):  # pylint: disable=too-few-public-methods,protected-access
    """A transformer that computes lagged values of a given time-indexed variable.

    Args:
        inputCol: The column that will be used to derive lagged variables.
        n_months: Number of months that will be considered for lags.

    """

    n_months = Param(
        Params._dummy(),
        "n_months",
        "Number of months for lag computation.",
    )

    ref_date = Param(
        Params._dummy(),
        "ref_date",
        "A reference date, used to compute number of months between rows.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            inputCol=None,
            n_months=None,
            ref_date=dt.date(2014, 1, 1),
        )
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
            "months_from_ref",
            F.months_between("periode", "ref_date").cast(T.IntegerType()),
        )

        lag_window = (
            Window().partitionBy("siren").orderBy(F.col("months_from_ref").asc())
        )

        for n in n_months:
            output_col = f"{input_col}_lag{n}m"
            dataset = dataset.withColumn(
                output_col,
                F.lag(F.col(input_col), n).over(lag_window),
            )
        return dataset.drop("ref_date", "months_from_ref")


class DiffOperator(
    Transformer, HasInputCol
):  # pylint: disable=too-few-public-methods,protected-access
    """A transformer that computes the time evolution of a given time-indexed variable.

    This transformer creates a LagOperator under the hood if the required lagged
    variable is not found in the dataset.

    Args:
        inputCol: The column that will be used to derive the diff.
        n_months: Number of months that will be considered for the difference.

    """

    n_months = Param(
        Params._dummy(),
        "n_months",
        "Number of months for diff computation.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            inputCol=None,
            n_months=None,
        )
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
                (F.col(f"{input_col}") - F.col(f"{input_col}_lag{n}m")),
            )

        return dataset.drop(*[f"{input_col}_lag{n}m" for n in missing_lags])


class TargetVariable(
    Transformer, HasInputCol, HasOutputCol
):  # pylint: disable=too-few-public-methods, protected-access
    """A transformer to compute the company failure target variable."""

    n_months = Param(
        Params._dummy(),
        "n_months",
        "Number of months for failure forecast.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(inputCol=None, outputCol=None, n_months=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCol (str): The column that will be used to derive target. It should
              contain the failure (judgment) date.
            outputCol (str): The new target variable column.
            n_months (int): Number of months that will be considered as target
              threshold.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame):
        """Create the learning target variable.

        Args:
            dataset: DataFrame to transform.

        Returns:
            Transformed DataFrame with an extra target column.

        """
        return dataset.withColumn(
            self.getOrDefault("outputCol"),
            (
                F.add_months(dataset["periode"], months=self.getOrDefault("n_months"))
                <= dataset[self.getOrDefault("inputCol")]
            ).cast(
                T.IntegerType()
            ),  # Pyspark models except integer or floating labels.
        ).fillna(value={self.getOrDefault("outputCol"): 0})


class PrivateCompanyFilter(Transformer):  # pylint: disable=too-few-public-methods
    """A transformer that filters a dataset according to its public/private nature."""

    def _transform(  # pylint: disable=no-self-use
        self, dataset: pyspark.sql.DataFrame
    ) -> pyspark.sql.DataFrame:
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


class HasPaydexFilter(Transformer):  # pylint: disable=too-few-public-methods
    """A transformer that filters according to paydex data availability."""

    def _transform(  # pylint: disable=no-self-use
        self, dataset: pyspark.sql.DataFrame
    ) -> pyspark.sql.DataFrame:
        """Filters out samples that do not have paydex data.

        Args:
            dataset: DataFrame to filter.

        Returns:
            Filtered DataFrame.

        """
        return dataset.filter(
            F.col("paydex").isNotNull() & F.col("paydex_diff12m").isNotNull()
        )


class WorkforceFilter(Transformer):  # pylint: disable=too-few-public-methods
    """A transformer to filter the dataset according to workforce size."""

    def _transform(  # pylint: disable=no-self-use
        self, dataset: pyspark.sql.DataFrame
    ) -> pyspark.sql.DataFrame:
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


class LinearInterpolationOperator(
    Transformer, HasInputCols
):  # pylint: disable=too-few-public-methods,protected-access
    """A transformer that fills missing values using linear interpolation.

    Data is grouped using the `id_cols` columns and ordered using the `time_col`, any
    null values gap between non-null values will be filled.

    Args :
        inputCols: Columns to filled.
        id_cols: Entity index, along which the dataset will be partitioned.
        time_col: Time index, used to sort the dataset.

    """

    id_cols = Param(
        Params._dummy(),
        "id_cols",
        "Id columns to group for interpolation.",
    )
    time_col = Param(
        Params._dummy(),
        "time_col",
        "Columns to follow for interpolation.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(id_cols="siren", time_col="periode", inputCols=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            inputCols (list[str]): Columns to fill.
            id_cols (str or list[str]): Entity index, along which the dataset will be
              partitioned.
            time_col (str): Time index, used to sort the dataset.

        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Use linear interpolation to fill missing time-series values.

        Args:
            dataset: DataFrame with time-series columns containing missing values.

        Returns:
            DataFrame where time-series missing values are filled through interpolation.

        """
        id_cols: Union[str, List[str]] = self.getOrDefault("id_cols")
        time_col: str = self.getOrDefault("time_col")
        input_cols: List[str] = self.getOrDefault("inputCols")

        w = Window.partitionBy(id_cols).orderBy(time_col)
        w_start = (
            Window.partitionBy(id_cols)
            .orderBy(time_col)
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        w_end = (
            Window.partitionBy(id_cols)
            .orderBy(time_col)
            .rowsBetween(0, Window.unboundedFollowing)
        )
        output_df = dataset.withColumn("rn", F.row_number().over(w))

        for col in input_cols:

            # Assign index for non-null rows within partitioned windows
            output_df = output_df.withColumn(
                "rn_not_null", F.when(F.col(col).isNotNull(), F.col("rn"))
            )

            # Create relative references to the gap start value (left bound)
            output_df = output_df.withColumn(
                "left_bound_val", F.last(col, ignorenulls=True).over(w_start)
            )
            output_df = output_df.withColumn(
                "left_bound_rn", F.last("rn_not_null", ignorenulls=True).over(w_start)
            )

            # Create relative references to the gap end value (right bound)
            output_df = output_df.withColumn(
                "right_bound_val", F.first(col, ignorenulls=True).over(w_end)
            )
            output_df = output_df.withColumn(
                "right_bound_rn", F.first("rn_not_null", ignorenulls=True).over(w_end)
            )

            # Create references to gap length and current gap position.
            output_df = output_df.withColumn(
                "interval_length_rn", F.col("right_bound_rn") - F.col("left_bound_rn")
            )
            output_df = output_df.withColumn(
                "curr_rn",
                F.col("interval_length_rn") - (F.col("right_bound_rn") - F.col("rn")),
            )

            # Compute linear interpolation value
            lin_interp_col = F.col("left_bound_val") + (
                F.col("right_bound_val") - F.col("left_bound_val")
            ) / F.col("interval_length_rn") * F.col("curr_rn")
            output_df = output_df.withColumn(
                col,
                F.when(F.col(col).isNull(), lin_interp_col).otherwise(F.col(col)),
            )

        return output_df.drop(
            "rn",
            "rn_not_null",
            "left_bound_val",
            "right_bound_val",
            "left_bound_rn",
            "right_bound_rn",
            "interval_length_rn",
            "curr_rn",
        )


class RandomResampler(
    Transformer
):  # pylint: disable=too-few-public-methods,protected-access
    """Resample dataset to attain some specific (binary) classes ratio.

    The dataset is resampled according to the min_class_ratio parameter so as to obtain
    the requested balance between the two classes. If `method` is "undersampling", the
    majority class samples are subsampled, if `method` is "oversampling", the minority
    class samples are oversampled using random sampling with replacement.

    Args:
        seed: Sampling random seed.
        method: "undersampling" will delete majority class samples, while "oversampling"
          will sample with replacement from minority class.
        class_col: Class label column.
        min_class_ratio: Requested (minority class / dataset size) ratio.

    """

    seed = Param(
        Params._dummy(),
        "seed",
        "Sampling random seed.",
    )
    method = Param(
        Params._dummy(),
        "method",
        "Resampling method.",
    )
    min_class_ratio = Param(
        Params._dummy(),
        "min_class_ratio",
        "Requested (minority class / total n samples) output ratio.",
    )
    class_col = Param(
        Params._dummy(),
        "class_col",
        "Class label column.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            class_col="failure",
        )
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        Args:
            seed (int): Sampling random seed.
            method (str): "undersampling" will delete majority class samples, while
              "oversampling" will sample with replacement from minority class.
            class_col (str): Class label column.
            min_class_ratio (float): Requested (minority class / dataset size) ratio.

        """
        return self._set(**kwargs)

    def _transform(  # pylint:disable=too-many-locals
        self, dataset: pyspark.sql.DataFrame
    ) -> pyspark.sql.DataFrame:
        """Resamples the dataset.

        Returns:
            DataFrame, resampled according to requested ratio.

        """
        seed: int = self.getOrDefault("seed")
        class_col: str = self.getOrDefault("class_col")
        method: str = self.getOrDefault("method")
        min_class_ratio: float = self.getOrDefault("min_class_ratio")
        maj_class_ratio: float = 1.0 - min_class_ratio

        if method not in {"undersampling", "oversampling"}:
            raise ValueError(f"Unknown resampling method {method}.")

        # Get class counts and filter subsets
        class_counts = dataset.groupBy(class_col).count().rdd.collectAsMap()
        majority_class_label = max(class_counts, key=class_counts.get)
        minority_class_label = min(class_counts, key=class_counts.get)
        majority_class_count: int = class_counts[majority_class_label]
        minority_class_count: int = class_counts[minority_class_label]
        if len(class_counts) == 1:
            raise ValueError("There is only 1 class in the dataset.")
        majority_class_df = dataset.filter(F.col(class_col) == majority_class_label)
        minority_class_df = dataset.filter(F.col(class_col) == minority_class_label)
        if method == "undersampling":
            # We compute total number of samples in resampled dataset given fixed
            # minority class samples, then we undersample the majority class subset.
            subset_size = int(minority_class_count / min_class_ratio)
            n_post_sampling_maj_class = int(maj_class_ratio * subset_size)
            downsampled_df = majority_class_df.sample(
                withReplacement=False,
                fraction=n_post_sampling_maj_class / majority_class_count,
                seed=seed,
            )
            resampled_df = downsampled_df.union(minority_class_df)
        elif method == "oversampling":
            # We compute total number of samples in resampled dataset given fixed
            # majority class samples, then we oversample the minority class subset.
            subset_size = int(majority_class_count / maj_class_ratio)
            n_post_sampling_min_class = int(min_class_ratio * subset_size)
            upsampled_df = minority_class_df.sample(
                withReplacement=True,
                fraction=n_post_sampling_min_class / minority_class_count,
                seed=seed,
            )
            resampled_df = upsampled_df.union(majority_class_df)

        return resampled_df
