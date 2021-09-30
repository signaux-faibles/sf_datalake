import pyspark.sql.functions as F


def load_source(fpath, spl_size=None):
    df = spark.read.orc(fpath)
    if spl_size is not None:
        df = df.sample(spl_size)
    return df


def count_missing_values(df):
    """Counts number of nulls in each column, omitting columns of given type."""
    return df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])


def count_nan_values(df, omit_type=("timestamp", "string", "date", "bool")):
    """Counts number of nan values in float / double columns

    Columns of given type are omitted.
    """

    return df.select(
        [
            F.count(F.when(F.isnull(c), c)).alias(c)
            for (c, c_type) in df.dtypes
            if c_type not in omit_type
        ]
    )


def get_basic_statistics(df):
    n_rows = df.count()
    n_sirens_sf = df[["siren"]].distinct().count()
    n_sirens_dgfip = df[["siren_dgfip"]].distinct().count()
    n_columns = len(df.columns)

    print(f"Nb lignes: {n_rows}")
    print(f"Nb SIRENs sf: {n_sirens_sf}")
    print(f"Nb SIRENs DGFiP: {n_sirens_dgfip}")
    print(f"Nb columns: {n_columns}")
    return (n_rows, n_sirens_sf, n_sirens_dgfip, n_columns)


def get_full_statistics(df):
    (n_rows, n_sirens, _, n_columns) = get_basic_statistics(df)
    date_deb_exercice_span = df.select(
        min("date_deb_exercice"),
        max("date_deb_exercice"),
    ).first()
    date_fin_exercice_span = df.select(
        min("date_fin_exercice"),
        max("date_fin_exercice"),
    ).first()

    print(
        f"Date de début d'exercice s'étendent de \
        {date_deb_exercice_span[0].strftime('%d/%m/%Y')} à \
        {date_deb_exercice_span[1].strftime('%d/%m/%Y')}"
    )
    print(
        f"Date de fin d'exercice s'étendent de \
        {date_fin_exercice_span[0].strftime('%d/%m/%Y')} à \
        {date_fin_exercice_span[1].strftime('%d/%m/%Y')}"
    )
    return (n_rows, n_sirens, n_columns)


def acoss_make_avg_delta_dette_par_effectif(data):
    """
    Compute the average change in social debt / effectif
    Output column : 'avg_delta_dette_par_effectif'
    """

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

    columns_to_drop = ["dette_par_effectif", "dette_par_effectif_past_3"]
    data = data.drop(*columns_to_drop)  # , axis=1, inplace=True)
    return data


# from https://stackoverflow.com/questions/44627386/how-to-find-count-of-null-and-nan-values-for-each-column-in-a-pyspark-dataframe
# create function for checking nulls per column in a DataFrame
def count_missings(spark_df, sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select(
        [
            F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c)
            for (c, c_type) in spark_df.dtypes
            if c_type not in ("timestamp", "string", "date")
        ]
    ).toPandas()

    if len(df) == 0:
        print("There are no any missing values!")
        return None

    if sort:
        return df.rename(index={0: "count"}).T.sort_values("count", ascending=False)

    return df


def oversample(df):
    # Implement oversampling method
    # calculate ratio
    major_df = df.filter(df.label == 0)
    minor_df = df.filter(df.label == 1)
    ratio = int(major_df.count() / minor_df.count())
    a = range(ratio)

    # duplicate the minority rows
    oversampled_df = minor_df.withColumn(
        "dummy", F.explode(F.array([F.lit(x) for x in a]))
    ).drop("dummy")

    # combine both oversampled minority rows and previous majority rows
    combined_df = major_df.unionAll(oversampled_df)
    return combined_df


def evaluate(results):
    # evaluate results
    correct_count = results.filter(results.label == results.prediction).count()
    total_count = results.count()

    correct_1_count = results.filter(
        (results.label == 1) & (results.prediction == 1)
    ).count()
    total_1_test = results.filter((results.label == 1)).count()
    total_1_predict = results.filter((results.prediction == 1)).count()

    print("All correct predections count: ", correct_count)
    print("Total count: ", total_count)
    print("Accuracy %: ", (correct_count / total_count) * 100)
    print("Recall %: ", (correct_1_count / total_1_test) * 100)
    print("Precision %: ", (correct_1_count / total_1_predict) * 100)
