"""Transformer class.

Transformer is an abstract class to define different transformers.
"""

import abc
from typing import Iterable, Optional

import pyspark.sql  # pylint: disable=E0401
from pyspark.ml.feature import StandardScaler, VectorAssembler  # pylint: disable=E0401


class Transformer(abc.ABC):
    """Abstract class as an interface for subclasses that represent different transformers.
    A transformer handles data treatments to apply on multiple variables such as scaling
    before feeding data to a model. It also contains static methods useful for subclasses."""

    def __init__(self, config: dict, sampling_ratio: Optional[float] = None):
        """The class constructor. self.model is set to None by the constructor and will get a
        value once self.fit() is called.

        Args:
            config: the config parameters (see config.get_config())
            sampling_ratio: If desired, only a proportion (float smaller than 1) of the
                        dataset can be used for fitting in order to increase speed. Defaults
                        to None.
        """
        self.config = config
        self.colname_to_transform = "assembled_features"
        self.colname_transformed = "assembled_transformed_features"
        self.sampling_ratio = sampling_ratio
        self.assembler = VectorAssembler(
            inputCols=self.config["STD_SCALE_FEATURES"],
            outputCol=self.colname_to_transform,
        )
        self.model = None

    @abc.abstractmethod
    def fit(self, df: pyspark.sql.DataFrame):
        """Fit a Tranformer on some pre-assembled data.

        Args:
            df : The DataFrame to fit on .
        """

    @abc.abstractmethod
    def transform(
        self,
        df: pyspark.sql.DataFrame,
        label_col: str,
        keep_cols: Optional[Iterable[str]] = None,
    ) -> pyspark.sql.DataFrame:
        """Transform the data according to the fitted transformer.

        Args:
            df: The spark dataframe to transform.
            label_col: The name of the column containing the target variable.
                    It should be a column of booleans.
            keep_cols: If some columns, other than those inside features and label,
                    are to be preserved after scaling, they should be mentioned here as a list.

        Returns:
            The transformed DataFrame.
        """


class StandardTransformer(Transformer):
    """Transformer that implements the standard scaling."""

    def fit(self, df: pyspark.sql.DataFrame):  # pylint: disable=C0115
        assembled_to_fit = self.assembler.transform(df)
        assembled_to_fit = (
            assembled_to_fit.sample(self.sampling_ratio)
            if self.sampling_ratio is not None
            else assembled_to_fit
        )

        scaler = StandardScaler(
            # withMean=True,
            withStd=True,
            inputCol=self.colname_to_transform,
            outputCol=self.colname_transformed,
        )

        self.model = scaler.fit(assembled_to_fit)

    def transform(
        self,
        df: pyspark.sql.DataFrame,
        label_col: str,
        keep_cols: Optional[Iterable[str]] = None,
    ) -> pyspark.sql.DataFrame:  # pylint: disable=C0115

        assembled_df = self.assembler.transform(df)
        scaled_data = self.model.transform(assembled_df)

        selected_cols = [self.colname_transformed, label_col]
        if keep_cols is not None:
            selected_cols += keep_cols
        selected_data = (
            scaled_data.select(selected_cols)
            .withColumnRenamed(self.colname_transformed, "features")
            .withColumn("label", df[label_col].astype("integer"))
        )
        return selected_data
