"""Transformer class.

Transformer is an abstract class to build different transformers.
A transformer handles data treatments to apply on multiple variables
such as scaling before feeding data to a model.
"""
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import pyspark  # pylint: disable=E0401
import pyspark.sql  # pylint: disable=E0401
from pyspark.ml.feature import StandardScaler, VectorAssembler  # pylint: disable=E0401

from sf_datalake.config import Config


class Transformer(ABC):  # pylint: disable=C0115
    def __init__(self, config: Config):
        self.config = config.data
        self.model = None

    @abstractmethod
    def fit(  # pylint: disable=R0913
        self,
        df: pyspark.sql.DataFrame,
        scaler_type: str,
        input_colname: str,
        output_colname: str,
        sampling_ratio: Optional[float] = None,
    ):
        """Fit a transformer."""

    @abstractmethod
    def transform(
        self,
        df: pyspark.sql.DataFrame,
        features_col: str,
        label_col: str,
        keep_cols: Optional[Iterable[str]] = None,
    ) -> pyspark.sql.DataFrame:
        """Transform the data according to the fitted transformer.

        Args:
            df: The spark dataframe to transform.
            features_col: The name of the features column (see assemble_features()).
            label_col: The name of the column containing the target variable.
            It should be a column of booleans.
            keep_cols: If some columns, other than those inside features and label,
            are to be preserved after scaling, they should be mentioned here as a list.

        Returns:
            The transformed DataFrame.

        """

    @abstractmethod
    def run(
        self,
        data_train: pyspark.sql.DataFrame,
        data_test: pyspark.sql.DataFrame,
        data_prediction: pyspark.sql.DataFrame,
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
        """Run the transform function on multiple DataFrames (train, test, prediction).

        Returns:
            For each DataFrame, the transformed DataFrame.
        """

    @staticmethod
    def assemble_features(
        df: pyspark.sql.DataFrame,
        features_col: Iterable[str],
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


class BaseTransformer(Transformer):
    """Transformer that implements the standard scaling."""

    def fit(  # pylint: disable=R0913
        self,
        df: pyspark.sql.DataFrame,
        scaler_type: str,
        input_colname: str,
        output_colname: str,
        sampling_ratio: Optional[float] = None,
    ):
        """Creates and fits a scaler object to some pre-assembled data.

        Args:
            df: The DataFrame to fit on.
            scaler_type: The desired scaler type. Currently only supports `standard`.
            input_colname: The name of the input assembled features column (see
                        assemble_features()).
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
            raise NotImplementedError(
                "We'll have to wait for a newer version of pyspark."
            )
            # scaler = RobustScaler(
            #     inputCol=input_colname,
            #     outputCol=output_colname,
            # )
        else:
            raise NameError(f"Unkown scaler type {scaler_type}.")
        assembled_to_fit = Transformer.assemble_features(
            df, self.config["STD_SCALE_FEATURES"], input_colname
        )
        assembled_to_fit = (
            assembled_to_fit.sample(sampling_ratio)
            if sampling_ratio is not None
            else assembled_to_fit
        )
        self.model = scaler.fit(assembled_to_fit)

    def transform(
        self,
        df: pyspark.sql.DataFrame,
        features_col: str,
        label_col: str,
        keep_cols: Optional[Iterable[str]] = None,
    ) -> pyspark.sql.DataFrame:
        """Scales data using a pre-fitted scaler.

        Args:
            df: The spark dataframe to scale.
            features_col: The name of the features column (see assemble_features()).
            label_col: The name of the column containing the target variable.
            It should be a column of booleans.
            keep_cols: If some columns, other than those inside features and label,
            are to be preserved after scaling, they should be mentioned here as a list.

        Returns:
            The scaled DataFrame.

        """
        assembled_df = Transformer.assemble_features(
            df, self.config["STD_SCALE_FEATURES"], "assembled_std_features"
        )
        scaled_data = self.model.transform(assembled_df)

        selected_cols = [features_col, label_col]
        if keep_cols is not None:
            selected_cols += keep_cols
        selected_data = (
            scaled_data.select(selected_cols)
            .withColumnRenamed(features_col, "features")
            .withColumn("label", df[label_col].astype("integer"))
        )
        return selected_data

    def run(
        self, data_train, data_test, data_prediction
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
        self.fit(
            data_train,
            scaler_type="standard",
            input_colname="assembled_std_features",
            output_colname="std_scaled_features",
        )
        scaled_train = self.transform(
            df=data_train,
            features_col="std_scaled_features",
            label_col="failure_within_18m",
        )
        scaled_test = self.transform(
            df=data_test,
            features_col="std_scaled_features",
            label_col="failure_within_18m",
            keep_cols=["siren", "time_til_failure"],
        )
        scaled_prediction = self.transform(
            df=data_prediction,
            features_col="std_scaled_features",
            label_col="failure_within_18m",
            keep_cols=["siren"],
        )
        return scaled_train, scaled_test, scaled_prediction


def factory_transformer(config: Config) -> Transformer:
    """Factory for transformers."""
    transformers = {"BaseTransformer": BaseTransformer}
    return transformers[config.get_config()["TRANSFORMER"]]
