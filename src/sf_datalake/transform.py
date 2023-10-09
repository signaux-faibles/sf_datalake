"""Utilities and classes for handling and transforming datasets."""

import datetime as dt
import itertools
from typing import List, Union

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

    Uses pyspark.sql.DataFrame.fillna or an (statistical) Imputer object to fill missing
    values. Both strategies are mutually exclusive, so use either `value` or
    `stat_strategy`.

    Args:
      inputCols: The input dataset columns to consider for filling.
      value: Value to replace null values with. It must be a mapping from column name
        (string) to replacement value. The replacement value must be an int, float,
        boolean, or string.
      stat_strategy : strategy for the Imputer. Possible values are : 'mean', 'median' \
        and 'mode'

    """

    value = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "value",
        "Value to replace null values with.",
    )
    stat_strategy = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "stat_strategy",
        "Statistic method to use into the Imputer class.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(value=None, stat_strategy=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this transformer.

        inputCols (list[str]): The input dataset columns to consider for filling.
        value (dict): Value to replace null values with. It must be a mapping from
          column name (string) to replacement value. If it is None, stat imputaion is
          applied.
        stat_strategy (string) : strategy for the Imputer class. Possible values are:
          'mean', 'median' and 'mode'.
        """
        return self._set(**kwargs)

    def _transform(self, dataset: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Fills entries containing missing values.

        Args:
            dataset: DataFrame to transform containing missing values.

        Returns:

            DataFrame where previously missing values are either filled.

        """
        stat_strategy: str = self.getOrDefault("stat_strategy")
        value: dict = self.getOrDefault("value")
        input_cols: List[str] = self.getOrDefault("inputCols")

        if value is not None and stat_strategy is not None:
            raise ValueError(
                "`value` and `stat_strategy` are mutually exclusive. Use either one."
            )
        if value is not None:
            dataset = dataset.fillna(
                {var: val for var, val in value.items() if var in input_cols}
            )
        elif stat_strategy is not None:
            dtypes = [dtype for _, dtype in dataset.select(input_cols).dtypes]
            if any(dtype in {"bool", "timestamp", "string"} for dtype in dtypes):
                raise ValueError(
                    "Statistical imputation of a non-numerical variable is not "
                    "supported."
                )
            imputer = Imputer(strategy=stat_strategy)
            imputer.setInputCols(input_cols)
            imputer.setOutputCols(input_cols)
            dataset = imputer.fit(dataset).transform(dataset)
        else:
            raise ValueError("Either `value` or `stat_strategy` must be set.")
        return dataset


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


class MovingAverage(Transformer, HasInputCol):  # pylint: disable=too-few-public-methods
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


class LagOperator(Transformer, HasInputCol):  # pylint: disable=too-few-public-methods
    """A transformer that computes lagged values of a given time-indexed variable.

    A forward or backward fill can optionally be performed when lag data is unavailable.
    Both are mutually exclusive.

    Args:
        inputCol: The column that will be used to derive lagged variables.
        n_months: Number of months that will be considered for lags.
        bfill: If set, performs a backward completion on missing lag data.
        ffill: If set, performs a forward completion on missing lag data.

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

    bfill = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "bfill",
        "A boolean, used to specify if a backward completion is applied to missing lag \
        data.",
    )

    ffill = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "ffill",
        "A boolean, used to specify if a forward completion is applied to missing lag \
        data.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            inputCol=None,
            n_months=None,
            bfill=False,
            ffill=False,
            ref_date=dt.date(2014, 1, 1),
        )
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this LagOperator transformer.

        Args:
            inputCol (str): The column that will be used to derive lagged variables.
            n_months (int or list): Number of months that will be considered for lags.
            bfill (bool) : If set, performs a backward completion on missing lag data.
            ffill (bool) : If set, performs a forward completion on missing lag data.

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
        bfill = self.getOrDefault("bfill")
        ffill = self.getOrDefault("ffill")
        if isinstance(n_months, int):
            n_months = [n_months]
        elif isinstance(n_months, list):
            pass
        else:
            raise ValueError("`n_months` should either be an int or a list of ints.")

        if bfill and ffill:
            raise ValueError("`ffill` and `bfill` are mutually exclusive.")

        dataset = dataset.withColumn(
            "ref_date", F.lit(self.getOrDefault("ref_date"))
        ).withColumn(
            "months_from_ref",
            F.months_between("periode", "ref_date").cast(T.IntegerType()),
        )

        lag_window = (
            Window().partitionBy("siren").orderBy(F.col("months_from_ref").asc())
        )
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

        for n in n_months:
            output_col = f"{input_col}_lag{n}m"
            dataset = dataset.withColumn(
                output_col,
                F.lag(F.col(input_col), n).over(lag_window),
            )
            # Fill missing values
            if ffill or bfill:
                # When we do a backward fill, we use a forward_window: we're looking for
                # future (lagged) values from periods of time where lag values were
                # unavailable.
                fill_window = forward_window if bfill else backward_window
                lookup_function = F.first if bfill else F.last
                dataset = dataset.withColumn(
                    output_col,
                    F.when(
                        # If not all computed lagged values are null, use them for
                        # filling.
                        F.count(F.when(F.isnull(output_col), output_col)).over(
                            fill_window
                        )
                        != F.count(F.col(output_col)).over(fill_window),
                        lookup_function(output_col, ignorenulls=True).over(fill_window),
                    ).otherwise(
                        # Otherwise, use the non-lagged input column values.
                        lookup_function(input_col, ignorenulls=True).over(fill_window)
                    ),
                )

        return dataset.drop("ref_date", "months_from_ref")


class DiffOperator(Transformer, HasInputCol):  # pylint: disable=too-few-public-methods
    """A transformer that computes the time evolution of a given time-indexed variable.

    This transformer creates a LagOperator under the hood if the required lagged
    variable is not found in the dataset. The output(s) can either be a difference,
    or a slope, computed as the ratio of the slope over the duration, i.e. the
    `n_months` parameter.

    Args:
        inputCol: The column that will be used to derive the diff.
        n_months: Number of months that will be considered for the difference.
        slope: If True, divide the computed difference by its duration in months.
        bfill: If set, performs a backward completion on missing lag data.
        ffill: If set, performs a forward completion on missing lag data.

    """

    slope = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "slope",
        "Divide difference by the duration.",
    )
    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "n_months",
        "Number of months for diff computation.",
    )
    bfill = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "bfill",
        "A boolean, used to specify if a backward completion is applied to missing lag \
        data.",
    )

    ffill = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "ffill",
        "A boolean, used to specify if a forward completion is applied to missing lag \
        data.",
    )

    @keyword_only
    def __init__(self, **kwargs):
        super().__init__()
        self._setDefault(
            inputCol=None,
            n_months=None,
            slope=False,
            bfill=False,
            ffill=False,
        )
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        """Set parameters for this LagOperator transformer.

        Args:
            inputCol (str): The column that will be used to derive lagged variables.
            n_months (int or list): Number of months that will be considered for lags.
            slope: If True, divide the computed difference by its duration in months.

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
        bfill = self.getOrDefault("bfill")
        ffill = self.getOrDefault("ffill")
        compute_slope = self.getOrDefault("slope")
        if isinstance(n_months, int):
            n_months = [n_months]
        elif isinstance(n_months, list):
            pass
        else:
            raise ValueError("`n_months` should either be an int or a list of ints.")
        var_coeff = [1 / n if compute_slope else 1 for n in n_months]
        var_name = "slope" if compute_slope else "diff"

        # Compute lagged variables if needed
        missing_lags = [
            n for n in n_months if f"{input_col}_lag{n}m" not in dataset.columns
        ]
        dataset = PipelineModel(
            [
                LagOperator(inputCol=input_col, n_months=n, bfill=bfill, ffill=ffill)
                for n in missing_lags
            ]
        ).transform(dataset)

        # Compute diffs
        for i, n in enumerate(n_months):
            dataset = dataset.withColumn(
                f"{input_col}_{var_name}{n}m",
                (F.col(f"{input_col}") - F.col(f"{input_col}_lag{n}m")) * var_coeff[i],
            )

        return dataset.drop(*[f"{input_col}_lag{n}m" for n in missing_lags])


class TargetVariable(
    Transformer, HasInputCol, HasOutputCol
):  # pylint: disable=too-few-public-methods
    """A transformer to compute the company failure target variable."""

    n_months = Param(
        Params._dummy(),  # pylint: disable=protected-access
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
):  # pylint: disable=too-few-public-methods
    """A transformer that fills missing values using linear interpolation.

    Data is grouped using the `id_cols` columns and ordered using the `time_col`, any
    null values gap between non-null values will be filled.

    Args :
        inputCols: Columns to filled.
        id_cols: Entity index, along which the dataset will be partitioned.
        time_col: Time index, used to sort the dataset.

    """

    id_cols = Param(
        Params._dummy(),  # pylint: disable=protected-access
        "id_cols",
        "Id columns to group for interpolation.",
    )
    time_col = Param(
        Params._dummy(),  # pylint: disable=protected-access
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
