import pyspark


def acoss_make_avg_delta_dette_par_effectif(data: pyspark.sql.DataFrame):
    """Compute the average change in social debt / number of employees.

    Args:
        data : the input data containing required social debt columns.

    Returns:
        A dataset containing a new 'avg_delta_dette_par_effectif' column.
    """

    data = data.withColumn(
        "dette_par_effectif",
        (data["montant_part_ouvriere"] + data["montant_part_patronale"])
        / data["effectif"],
    )

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

    columns_to_drop = ["dette_par_effectif", "dette_par_effectif_past_3"]
    data = data.drop(*columns_to_drop)
    return data
