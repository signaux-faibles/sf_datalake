"""Model class.

Model is an abstract class to define different models.
"""

import abc
from typing import Tuple

import pyspark.sql  # pylint: disable=E0401
from pyspark import ml  # pylint: disable=E0401
from pyspark.sql import functions as F  # pylint: disable=E0401
from pyspark.sql.types import FloatType, StringType  # pylint: disable=E0401


class Model(abc.ABC):
    """Abstract class as an interface for subclasses that represent different
    models. It also contains static methods useful for subclasses."""

    def __init__(self, config: dict):
        """The class constructor. self.model is set to None by the constructor
        and will get a value once self.fit() is called.

        Args:
            config: the config parameters (see config.get_config())
        """
        self.config = config
        self.model = None

    @abc.abstractmethod
    def fit(self, data_train: pyspark.sql.DataFrame):
        """Fit the model and store it in self.model.

        Args:
            data_train: train sample
        """

    @abc.abstractmethod
    def predict(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Compute the predictions on the prediction sample

        Args:
            df: prediction sample

        Returns:
            df with a new column 'probability' for the predictions
        """

    @abc.abstractmethod
    def explain(
        self, df: pyspark.sql.DataFrame
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
        """Compute the contribution of multiples groups of features and the TOP3
           contributions of the individual features.

        Args:
            df: prediction sample

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
    """Subclass of the abstractclass Model (see help(Model) for more details
    on the interface). This class defines a logistic regression model with
    regularization."""

    def fit(self, data_train):  # pylint: disable=C0115
        blor = ml.classification.LogisticRegression(
            regParam=self.config["REGULARIZATION_COEFF"],
            standardization=False,
            maxIter=self.config["MAX_ITER"],
            tol=self.config["TOL"],
        )
        self.model = blor.fit(data_train)

    def predict(self, df) -> pyspark.sql.DataFrame:  # pylint: disable=C0115
        df_predicted = (
            self.model.transform(df)
            .select(["siren", "time_til_failure", "label", "probability"])
            .withColumn(
                "probability", Model.positive_class_proba_extractor("probability")
            )
        )
        return df_predicted

    def explain(
        self, df
    ) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:  # pylint: disable=C0115
        # Explain prediction
        meso_features = [
            feature
            for features in self.config["FEATURE_GROUPS"].values()
            for feature in features
        ]

        # Get feature influence
        ep = (
            ml.feature.ElementwiseProduct()
        )  # [TODO] Will need so adjustments because this line needs a SparkContext initialized
        ep.setScalingVec(self.model.coefficients)
        ep.setInputCol("features")
        ep.setOutputCol("eprod")

        explanation_df = (
            ep.transform(df)
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
        macro_scores_df = explanation_df.select(
            ["siren"] + self.config["MACRO_SCORES_COLUMNS"]
        )
        return macro_scores_df, micro_scores_df
