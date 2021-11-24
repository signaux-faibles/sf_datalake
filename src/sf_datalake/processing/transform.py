"""Data processing before feeding into pyspark learning algorithms.
"""

from typing import Iterable

import pyspark  # pylint: disable=E0401
from pyspark.ml.feature import StandardScaler, VectorAssembler  # pylint: disable=E0401


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
        df: The DataFrame to fit on.
        scaler_type: The desired scaler type. Currently only supports `standard`.
        input_colname: The name of the input assembled features column (see assemble_features()).
        output_colname: The name of the output assembled features column after scaling.
        sampling_ratio: If desired, only a proportion (float smaller than 1) of the
          dataset can be used for fitting in order to increase speed.

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
        scaler_model: A fitted pyspark scaler model object (see fit_scaler()).
        df: The spark dataframe to fit.
        features_col: The name of the features column (see assemble_features()).
        label_col: The name of the column containing the target variable.
          It should be a column of booleans.
        keep_cols: If some columns, other than those inside features and label,
          are to be preserved after scaling, they should be mentioned here as a list.

    Returns:
        The scaled DataFrame.

    """
    scaled_data = scaler_model.transform(df)

    selected_cols = [features_col, label_col]
    if keep_cols is not None:
        selected_cols += keep_cols
    selected_data = (
        scaled_data.select(selected_cols)
        .withColumnRenamed(features_col, "features")
        .withColumn("label", df[label_col].astype("integer"))
    )
    return selected_data
