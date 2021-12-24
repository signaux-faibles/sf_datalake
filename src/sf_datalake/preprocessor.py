"""Preprocessor class.

Preprocessor is an abstract class to define different preprocessors.
"""

import abc
from functools import reduce
from typing import Iterable

import pyspark.ml
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window


class Preprocessor(abc.ABC):
    """Abstract class as an interface for subclasses that represent
    different preprocessors. A preprocessor handles all basic data
    treatments before feeding data to a model. It also contains static
    methods useful for subclasses."""

    def __init__(self, config: dict):
        """The class constructor.

        Args:
            config: the config parameters (see utils.get_config())
        """
        self.config = config

    @abc.abstractmethod
    def run(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute the treatments."""

    @staticmethod
    def avg_delta_debt_per_size(data: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Computes the average change in social debt / nb of employees.

        Args:
            data: A DataFrame containing debt and company size ("effectif") data.

        Returns:
            A DataFrame with an extra `avg_delta_dette_par_effectif` column.

        """
        # TODO check if montant_part_ouvriere, montant_part_patronale,
        # montant_part_ouvriere_past_3, montant_part_patronale_past_3 exists?
        data = data.withColumn(
            "dette_par_effectif",
            (data["montant_part_ouvriere"] + data["montant_part_patronale"])
            / data["effectif"],
        )
        # TODO replace([np.nan, np.inf, -np.inf], 0)

        data = data.withColumn(
            "dette_par_effectif_past_3",
            (
                data["montant_part_ouvriere_past_3"]
                + data["montant_part_patronale_past_3"]
            )
            / data["effectif"],
        )
        # TODO replace([np.nan, np.inf, -np.inf], 0)

        data = data.withColumn(
            "avg_delta_dette_par_effectif",
            (data["dette_par_effectif"] - data["dette_par_effectif_past_3"]) / 3,
        )

        drop_columns = ["dette_par_effectif", "dette_par_effectif_past_3"]
        return data.drop(*drop_columns)

    @staticmethod
    def make_paydex_yoy(data: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Computes a new column for the dataset containing the year-over-year

        Args:
            data: A DataFrame object with "paydex_nb_jours" and "paydex_nb_jours_past_12"
            columns.

        Returns:
            The DataFrame with a new "paydex_yoy" column.

        """
        # TODO check if paydex_nb_jours, paydex_nb_jours_past_12 exists?
        return data.withColumn(
            "paydex_yoy", data["paydex_nb_jours"] - data["paydex_nb_jours_past_12"]
        )

    @staticmethod
    def make_paydex_bins(
        data: pyspark.sql.DataFrame,
        input_col: str = "paydex_nb_jours",
        output_col: str = "paydex_bins",
        num_buckets: int = 6,
    ) -> pyspark.sql.DataFrame:
        """Cuts paydex number of days data into quantile bins.

        Args:
            data: A pyspark.sql.DataFrame object.
            input_col: The name of the input column containing number of late days.
            output_col: The name of the output binned data column.
            num_buckets: Number of bins.

        Returns:
            The DataFrame with a new "paydex_group" column.

        """
        qds = pyspark.ml.feature.QuantileDiscretizer(
            inputCol=input_col,
            outputCol=output_col,
            handleInvalid="error",
            numBuckets=num_buckets,
        )
        return qds.fit(data).transform(data)

    @staticmethod
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
            df = df.withColumn(
                name, F.to_date(F.col(name).cast(StringType()), "yyyyMMdd")
            )
        return df

    @staticmethod
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


class BasePreprocessor(Preprocessor):
    """The basic preprocessor proceeds to the following steps:
    - Fill missing values by the median or a specific value (see config['SF_DEFAULT_VALUES'])
    - Aggregate at a SIREN level by sum or average
    - Data featuring:
            o creating ratio of the average debt
            o creating the objective variable 'failure_within_18m'
    - Filtering out firms on 'effectif' and 'code_naf' variables such that:
      effectif >= 10 AND code_naf NOT IN ('O', 'P')
    """

    def run(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        ### Default values for missing data
        non_ratio_variables = list(
            filter(lambda x: x[:3] != "RTO", self.config["MRV_VARIABLES"])
        )
        ratio_variables = list(
            filter(lambda x: x[:3] == "RTO", self.config["MRV_VARIABLES"])
        )

        mrv_default_values = {v: 0.0 for v in non_ratio_variables}
        data_medians = reduce(
            lambda x, y: x + y, df.approxQuantile(ratio_variables, [0.5], 0.05)
        )
        for var, med in zip(ratio_variables, data_medians):
            mrv_default_values[var] = med

        default_data_values = dict(
            **mrv_default_values, **self.config["SF_DEFAULT_VALUES"]
        )

        if self.config["FILL_MISSING_VALUES"]:
            df = df.fillna(
                {k: v for (k, v) in default_data_values.items() if k in df.columns}
            )
        else:
            df = df.fillna(
                {
                    "time_til_failure": self.config["SF_DEFAULT_VALUES"][
                        "time_til_failure"
                    ],
                }
            )

        ### Aggregation at SIREN level.
        # Signaux faibles variables
        df_sf = df.select(
            *(
                set(
                    self.config["SUM_VARIABLES"]
                    + self.config["AVG_VARIABLES"]
                    + self.config["BASE_VARIABLES"]
                )
            )
        )

        # Sums
        gb_sum = df_sf.groupBy("siren", "periode").sum(*self.config["SUM_VARIABLES"])
        for col_name in self.config["SUM_VARIABLES"]:
            gb_sum = gb_sum.withColumnRenamed(f"sum({col_name})", col_name)

        # Averages
        gb_avg = df_sf.groupBy("siren", "periode").avg(*self.config["AVG_VARIABLES"])
        for col_name in self.config["AVG_VARIABLES"]:
            gb_avg = gb_avg.withColumnRenamed(f"avg({col_name})", col_name)

        ### TODO : ratio_dette_moyenne12m should be computed from the
        ### aggregated ratio_dette variable.
        # w = df_sf.groupBy("siren", F.window(df.periode - 365
        # days, "365 days")).avg("ratio_dette")

        # Joining grouped data
        df_sf = (
            df_sf.drop(
                *(set(self.config["SUM_VARIABLES"] + self.config["AVG_VARIABLES"]))
            )
            .join(gb_sum, on=["siren", "periode"])
            .join(gb_avg, on=["siren", "periode"])
        )

        ### Feature engineering
        # delta_dette_par_effectif
        df_sf = Preprocessor.avg_delta_debt_per_size(df_sf)

        # ratio_dette : real computation after sum
        df_sf = df_sf.withColumn(
            "ratio_dette",
            (df_sf.montant_part_ouvriere + df_sf.montant_part_patronale)
            / df_sf.cotisation_moy12m,
        )

        df_sf = df_sf.dropDuplicates(["siren", "periode"])

        # DGFIP Variables
        df_dgfip = df.select(
            *(set(self.config["MRV_VARIABLES"]) | {"siren", "periode"})
        ).dropDuplicates(["siren", "periode"])

        # Joining data
        df = df_sf.join(df_dgfip, on=["siren", "periode"])

        if self.config["FILL_MISSING_VALUES"]:
            df = df.fillna(
                {k: v for (k, v) in default_data_values.items() if k in df.columns}
            )
        else:
            df = df.dropna(subset=tuple(self.config["FEATURES"]))

        # Creating objective variable 'failure_within_18m'
        df = df.withColumn("failure_within_18m", df["time_til_failure"] <= 18)

        # Filtering out firms on 'effectif' and 'code_naf' variables.
        df = df.select(
            *(
                set(
                    self.config["BASE_VARIABLES"]
                    + self.config["FEATURES"]
                    + self.config["TARGET_VARIABLE"]
                )
            )
        ).filter("effectif >= 10 AND code_naf NOT IN ('O', 'P')")

        df = df.withColumn(
            "failure_within_18m", df.failure_within_18m.astype("integer")
        )  # Needed  for models
        return df
