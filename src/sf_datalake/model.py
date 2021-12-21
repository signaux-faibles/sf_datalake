"""Model class.

Model is an abstract class to define different models.
"""

from typing import Tuple

import pyspark.ml
import pyspark.ml.classification
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

from sf_datalake.transformer import FormatProbability


def generate_stage(config: dict) -> pyspark.ml.Model:
    """Generate stage related to Model. Ready to be
    included in a pyspark.ml.Pipeline.

    Args:
        config : the config parameters (see config.get_config())

    Returns:
        A prepared Model.
    """
    stages = []
    stages += [get_model_from_conf(config["MODEL"])]
    stages += [FormatProbability()]
    return stages


def get_model_from_conf(model_config: dict) -> pyspark.ml.Model:
    """Get a Model from its configuration.

    Args:
        model_config: Configuration of the Model

    Returns:
        The selected Model with prepared parameters
    """
    factory = {
        "LogisticRegression": pyspark.ml.classification.LogisticRegression(
            labelCol="failure_within_18m",
            regParam=model_config["REGULARIZATION_COEFF"],
            standardization=False,
            maxIter=model_config["MAX_ITER"],
            tol=model_config["TOL"],
        )
    }
    return factory[model_config["MODEL_NAME"]]


@F.udf(returnType=StringType())
def get_max_value_colname(meso_features, max_col):
    """Extract the column name where the nth highest value is found."""
    row = F.array(meso_features)
    for i, name in enumerate(meso_features):
        if row[i] == max_col:
            return name
    raise NameError(f"Could not find columns associated to {max_col}")


def explain(
    config: dict, model: pyspark.ml.Model, df: pyspark.sql.DataFrame
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Compute the contribution of multiples groups of features and
       individual features depending of the model used.

    Args:
        config: the config parameters (see config.get_config())
        model: the model fit in the pipeline
        df: prediction sample

    Returns:
        - first, a DataFrame with the contribution of groups of features (macro)
        - second, a DataFrame with the TOP3 contributions of the features (micro)
    """
    factory = {"LogisticRegression": explain_LogisticRegression}
    return factory[config["MODEL"]["MODEL_NAME"]](config, model, df)


def explain_LogisticRegression(
    config: dict, model: pyspark.ml.Model, df: pyspark.sql.DataFrame
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Compute the contribution of multiples groups of features and the TOP3
           contributions of the individual features.

    Args:
        model: the model fit in the pipeline
        df: prediction sample
        config: the config parameters (see config.get_config())

    Returns:
        - first, a DataFrame with the contribution of groups of features (macro)
        - second, a DataFrame with the TOP3 contributions of the features (micro)
    """
    # Explain prediction
    meso_features = [
        feature
        for features in config["FEATURE_GROUPS"].values()
        for feature in features
    ]

    # Get feature influence
    ep = (
        pyspark.ml.feature.ElementwiseProduct()
    )  # [TODO] Will need so adjustments because this line needs a SparkContext initialized
    ep.setScalingVec(model.coefficients)
    ep.setInputCol("features")
    ep.setOutputCol("eprod")

    explanation_df = (
        ep.transform(df)
        .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
        .toDF(["siren"] + config["STD_SCALE_FEATURES"])
    )
    for group, features in config["MESO_URSSAF_GROUPS"].items():
        explanation_df = explanation_df.withColumn(
            group, sum(explanation_df[col] for col in features)
        ).drop(*features)

    # 'Macro' scores per group
    for group, features in config["FEATURE_GROUPS"].items():
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
            get_max_value_colname(meso_features, "1st_concerning_val"),
        )
        .withColumn(
            "2nd_concerning_feat",
            get_max_value_colname(meso_features, "2nd_concerning_val"),
        )
        .withColumn(
            "3rd_concerning_feat",
            get_max_value_colname(meso_features, "3rd_concerning_val"),
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
    macro_scores_df = explanation_df.select(["siren"] + config["MACRO_SCORES_COLUMNS"])
    return macro_scores_df, micro_scores_df
