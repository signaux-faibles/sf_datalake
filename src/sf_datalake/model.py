"""Model class.

Model is an abstract class to build different model.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pyspark.sql  # pylint: disable=E0401
from pyspark.ml.classification import LogisticRegression  # pylint: disable=E0401
from pyspark.ml.feature import ElementwiseProduct  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401
from pyspark.sql.types import FloatType, StringType  # pylint: disable=E0401

from sf_datalake.config import Config


class Model(ABC):  # pylint: disable=C0115
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.model = None

    @abstractmethod
    def fit(
        self, data_train: pyspark.sql.DataFrame, data_test: pyspark.sql.DataFrame
    ) -> pyspark.sql.DataFrame:
        """Fit the model and compute the prediction on the test sample.

        Args:
            data_train: train sample
            data_test: test sample

        Returns:
            data_test with a new column 'probability' for the predictions
        """

    @abstractmethod
    def predict(self, data_prediction: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute the predictions on the prediction sample

        Args:
            data_prediction: prediction sample

        Returns:
            data_prediction with a new column 'probability' for the predictions
        """

    @abstractmethod
    def explain(
        self, data_prediction: pyspark.sql.DataFrame
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
        """Compute the contribution of multiples groups of features and the TOP3
           contributions of the individual features.

        Args:
            data_prediction: prediction sample

        Returns:
            - first, a DataFrame with the contribution of groups of features (macro)
            - second, a DataFrame with the TOP3 contributions of the features (micro)
        """

    @staticmethod
    @F.udf(returnType=FloatType())
    def positive_class_proba_extractor(v):
        """Extract the probability as float."""
        return float(v[1])

    @staticmethod
    @F.udf(returnType=StringType())
    def get_max_value_colname(meso_features, max_col):
        """Extract the column name where the nth highest value is found."""
        row = F.array(meso_features)
        for i, name in enumerate(meso_features):
            if row[i] == max_col:
                return name
        raise NameError(f"Could not find columns associated to {max_col}")


class LogisticRegressionModel(Model):
    """A logistic regression model with regularization."""

    def fit(self, data_train, data_test):  # pylint: disable=C0115
        # Training
        logging.info(
            "Training logistic regression model with regularization \
            %.3f and %d iterations (maximum).",
            self.config["REGULARIZATION_COEFF"],
            self.config["MAX_ITER"],
        )
        blor = LogisticRegression(
            regParam=self.config["REGULARIZATION_COEFF"],
            standardization=False,
            maxIter=self.config["MAX_ITER"],
            tol=self.config["TOL"],
        )
        self.model = blor.fit(data_train)
        logging.info("Model weights: %.3f", self.model.coefficients)
        logging.info("Model intercept: %.3f", self.model.intercept)

        # Test data for optimal threshold computation
        logging.info("Running model on test dataset.")

        test_data = self.model.transform(data_test)
        test_data = test_data.select(
            ["siren", "time_til_failure", "label", "probability"]
        )
        test_data = test_data.withColumn(
            "probability", Model.positive_class_proba_extractor("probability")
        )
        return test_data

    def predict(self, data_prediction):  # pylint: disable=C0115
        logging.info("Running model on prediction dataset.")
        data_predicted = (
            self.model.transform(data_prediction)
            .drop(
                "prediction",
                "rawPrediction",
                "label",
            )
            .withColumn(
                "probability", Model.positive_class_proba_extractor("probability")
            )
        )
        return data_predicted

    def explain(self, data_prediction):  # pylint: disable=C0115
        # Explain prediction
        meso_features = [
            feature
            for features in self.config["FEATURE_GROUPS"].values()
            for feature in features
        ]

        # Get feature influence
        ep = ElementwiseProduct()
        ep.setScalingVec(self.model.coefficients)
        ep.setInputCol("features")
        ep.setOutputCol("eprod")

        explanation_df = (
            ep.transform(data_prediction)
            .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
            .toDF(["siren"] + self.config["STD_SCALE_FEATURES"])
        )
        for group, features in self.config["MESO_URSSAF_GROUPS"].items():
            explanation_df = explanation_df.withColumn(
                group, sum(explanation_df[col] for col in features)
            ).drop(*features)

        # 'Macro' scores per group
        for group, features in self.config["FEATURE_GROUPS"].items():
            explanation_df = explanation_df.withColumn(
                f"{group}_macro_score",
                1 / (1 + F.exp(-(sum(explanation_df[col] for col in features)))),
            )
        macro_scores_columns = [
            "sante_financiere_macro_score",
            "dette_urssaf_macro_score",
            "activite_partielle_macro_score",
            "misc_macro_score",
        ]

        # 'Micro' scores
        explanation_df = (
            explanation_df.withColumn(
                "1st_concerning_val",
                F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[0],
            )
            .withColumn(
                "2nd_concerning_val",
                F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[1],
            )
            .withColumn(
                "3rd_concerning_val",
                F.sort_array(F.array([F.col(x) for x in meso_features]), asc=False)[2],
            )
        )

        explanation_df = (
            explanation_df.withColumn(
                "1st_concerning_feat",
                Model.get_max_value_colname(meso_features, "1st_concerning_val"),
            )
            .withColumn(
                "2nd_concerning_feat",
                Model.get_max_value_colname(meso_features, "2nd_concerning_val"),
            )
            .withColumn(
                "3rd_concerning_feat",
                Model.get_max_value_colname(meso_features, "3rd_concerning_val"),
            )
        )
        micro_scores_columns = [
            "1st_concerning_val",
            "2nd_concerning_val",
            "3rd_concerning_val",
            "1st_concerning_feat",
            "2nd_concerning_feat",
            "3rd_concerning_feat",
        ]

        micro_scores_df = explanation_df.select(["siren"] + micro_scores_columns)
        macro_scores_df = explanation_df.select(["siren"] + macro_scores_columns)
        return macro_scores_df, micro_scores_df
