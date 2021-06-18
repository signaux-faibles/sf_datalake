import pyspark.sql.functions as F


def acoss_make_avg_delta_dette_par_effectif(data):
    """
    Compute the average change in social debt / effectif
    Output column : 'avg_delta_dette_par_effectif'
    """

    data = data.withColumn("dette_par_effectif", (data["montant_part_ouvriere"] + data["montant_part_patronale"]) / data["effectif"])
    # TODO replace([np.nan, np.inf, -np.inf], 0)

    data = data.withColumn("dette_par_effectif_past_3", (data["montant_part_ouvriere_past_3"] + data["montant_part_patronale_past_3"]) / data["effectif"])
    # TODO replace([np.nan, np.inf, -np.inf], 0)

    data = data.withColumn("avg_delta_dette_par_effectif", (data["dette_par_effectif"] - data["dette_par_effectif_past_3"]) / 3)
    
    columns_to_drop = ["dette_par_effectif", "dette_par_effectif_past_3"]
    data = data.drop(*columns_to_drop) # , axis=1, inplace=True)
    return data


# from https://stackoverflow.com/questions/44627386/how-to-find-count-of-null-and-nan-values-for-each-column-in-a-pyspark-dataframe
# create function for checking nulls per column in a DataFrame
def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(df) == 0:
        print("There are no any missing values!")
        return None

    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count",ascending=False)

    return df


def oversample(df):
    # Implement oversampling method
    # calculate ratio
    major_df = df.filter(df.label == 0)
    minor_df = df.filter(df.label == 1)
    ratio = int(major_df.count() / minor_df.count())
    a = range(ratio)

    # duplicate the minority rows
    oversampled_df = minor_df.withColumn("dummy", F.explode(F.array([F.lit(x) for x in a]))).drop('dummy')

    # combine both oversampled minority rows and previous majority rows 
    combined_df = major_df.unionAll(oversampled_df)
    return combined_df

def evaluate(results):
    # evaluate results
    correct_count = results.filter(results.label == results.prediction).count()
    total_count = results.count()

    correct_1_count = results.filter((results.label == 1) & (results.prediction == 1)).count()
    total_1_test = results.filter((results.label == 1)).count()
    total_1_predict = results.filter((results.prediction == 1)).count()

    print("All correct predections count: ", correct_count)
    print("Total count: ", total_count)
    print("Accuracy %: ", (correct_count / total_count)*100)
    print("Recall %: ", (correct_1_count / total_1_test)*100)
    print("Precision %: ", (correct_1_count / total_1_predict)*100)
