"""Feature engineering functions.
"""

from typing import List

import pyspark.ml  # pylint: disable=E0401
import pyspark.sql  # pylint: disable=E0401
import pyspark.sql.functions as F  # pylint: disable=E0401
from pyspark.sql.types import StringType  # pylint: disable=E0401
from pyspark.sql.window import Window  # pylint: disable=E0401


# Debt
def avg_delta_debt_per_size(data: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Compute the average change in social debt / nb of employees.

    Args:
      - data : A DataFrame containing debt and company size ("effectif") data.

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
        (data["montant_part_ouvriere_past_3"] + data["montant_part_patronale_past_3"])
        / data["effectif"],
    )
    # TODO replace([np.nan, np.inf, -np.inf], 0)

    data = data.withColumn(
        "avg_delta_dette_par_effectif",
        (data["dette_par_effectif"] - data["dette_par_effectif_past_3"]) / 3,
    )

    drop_columns = ["dette_par_effectif", "dette_par_effectif_past_3"]
    return data.drop(*drop_columns)


# Paydex
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


def make_paydex_bins(
    data: pyspark.sql.DataFrame,
    input_col: str = "paydex_nb_jours",
    output_col: str = "paydex_bins",
    num_buckets: int = 6,
) -> pyspark.sql.DataFrame:
    """Cut paydex number of days data into quantile bins.

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


def parse_date(df: pyspark.sql.DataFrame, colnames: List[str]) -> pyspark.sql.DataFrame:
    """Parse multiple columns of a pyspark.sql.DataFrame as date. Parsing is done inplace.

    Args:
        df (pyspark.sql.DataFrame)
        colnames (List[str]): Names of the columns

    Returns:
        pyspark.sql.DataFrame
    """

    for name in colnames:
        df = df.withColumn(name, F.to_date(F.col(name).cast(StringType()), "yyyyMMdd"))
    return df


def process_paiement(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Compute the featuring on 'paiement' data by adding new colunms.

    Args:
        df (pyspark.sql.DataFrame)

    Returns:
        pyspark.sql.DataFrame
    """
    df = df.withColumn("mvt_djc_int", F.unix_timestamp(F.col("mvt_djc")))
    df = df.orderBy("frp", "art_cleart", "mvt_djc").groupBy(
        ["frp", "art_cleart", "mvt_deff"]
    )
    df = df.agg(F.min("mvt_djc_int"), F.sum("mvt_mcrd"))
    df = df.select(["frp", "art_cleart", "min(mvt_djc_int)", "sum(mvt_mcrd)"])
    df = df.withColumnRenamed("min(mvt_djc_int)", "min_mvt_djc_int")
    df = df.withColumnRenamed("sum(mvt_mcrd)", "sum_mvt_mcrd")
    df = df.dropDuplicates()

    windowval = (
        Window.partitionBy("art_cleart")
        .orderBy(["frp", "min_mvt_djc_int"])
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    df = df.filter("sum_mvt_mcrd != 0").withColumn(
        "mnt_paiement_cum", F.sum("sum_mvt_mcrd").over(windowval)
    )
    df = df.withColumn("nb_paiement", F.count("sum_mvt_mcrd").over(windowval))
    df = df.dropDuplicates()
    return df
