"""Data processing before feeding into pyspark learning algorithms.
"""

from typing import Iterable

import pyspark
from pyspark.ml.feature import StandardScaler, VectorAssembler


def assemble_features(
    df: pyspark.sql.DataFrame,
    features_col: Iterable,
    output_col: str = "assembled_features",
):
    """Assembles features columns into a DenseVector object.

    Args:
        df: The DataFrame containing features data.
        features_col: The features columns names.
        output_col: The name of the assembled features.

    Returns:
        A DataFrame with a new column of assembled features.
    """
    assembler = VectorAssembler(inputCols=features_col, outputCol=output_col)
    return assembler.transform(df)


def fit_scaler(
    df: pyspark.sql.DataFrame,
    scaler_type: str,
    input_colname: str,
    output_colname: str,
    sampling_ratio: float = None,
) -> pyspark.ml.Model:
    """Creates and fits a scaler object to some pre-assembled data.

    Args:
        df:
        scaler_type:
        input_colname:
        output_colname:
        sampling_ratio:

    Returns:
        A scaler object fitted using the input DataFrame.
    """
    if scaler_type == "standard":
        scaler = StandardScaler(
            # withMean=True,
            withStd=True,
            inputCol=input_colname,
            outputCol=output_colname,
        )
    elif scaler_type == "robust":
        raise NotImplementedError("We'll have to wait for a newer version of pyspark.")
        # scaler = RobustScaler(
        #     inputCol=input_colname,
        #     outputCol=output_colname,
        # )
    else:
        raise NameError(f"Unkown scaler type {scaler_type}.")
    to_fit = df.sample(sampling_ratio) if sampling_ratio is not None else df
    scaler_model = scaler.fit(to_fit)
    return scaler_model


def scale_df(
    scaler_model: pyspark.ml.Model,
    df: pyspark.sql.DataFrame,
    features_col: str,
    label_col: str,
    keep_cols: list = None,
) -> pyspark.sql.DataFrame:
    """Scales data using a pre-fitted scaler.

    Args:
        scaler_model:
        df:
        features_col:
        label_col:
        keep_cols:

    Returns:
        The scaled DataFrame.
    """
    scaled_data = scaler_model.transform(df)

    selected_cols = [features_col]
    if keep_cols is not None:
        selected_cols += keep_cols

    selected_data = (
        scaled_data.select(selected_cols)
        .withColumnRenamed("features", features_col)
        .withColumn("label", df[label_col].astype("integer"))
    )
    return selected_data
