"""Model utilities and classes."""

from typing import List, Tuple

import pyspark.ml
import pyspark.ml.classification
import pyspark.sql.functions as F
from pyspark.sql.types import StringType


def generate_stages(config: dict) -> List[pyspark.ml.Model]:
    """Generate stages associated with a given Model.

    The returned list is ready to be included into a pyspark.ml.Pipeline object.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        A prepared Model.

    """
    stages = [get_model_from_conf(config["MODEL"])]
    return stages


def get_model_from_conf(model_config: dict) -> pyspark.ml.Model:
    """Generates a Model object from a given configuration.

    Args:
        model_config: The Model configuration. The dict contains parameters that
          corresponds to some pyspark.ml.Model arguments.

    Returns:
        The selected Model instantiated using the input config parameters.

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
    return factory[model_config["NAME"]]


def explain(
    config: dict, pipeline_model: pyspark.ml.PipelineModel, df: pyspark.sql.DataFrame
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Computes the contribution of different features to the predicted output.

    May depend on used model type.

    Args:
        config: Model configuration, as loaded by utils.get_config().
        pipeline_model: A fitted pipeline.
        df: The prediction samples.

    Returns:
        A tuple consisting of:
        - a DataFrame with the contribution of features groups (macro).
        - a DataFrame with the top 3 contributing features (micro).

    """
    model = None
    for stage in pipeline_model.stages:
        if config["MODEL"]["NAME"] in repr(stage):
            model = stage
            break
    if model is None:
        raise ValueError(
            f"Model with name {config['MODEL']['NAME']} could not be found in "
            "pipeline stages."
        )

    features_lists = [
        stage.getInputCols()
        for stage in pipeline_model.stages
        if isinstance(stage, pyspark.ml.feature.VectorAssembler)
    ]
    # We drop the last element of features_lists which is a stage where an assembler
    # only concatenates previous assembled feature groups.
    features = [feat for flist in features_lists[:-1] for feat in flist]

    factory = {"LogisticRegression": explain_logistic_regression}
    return factory[config["MODEL"]["NAME"]](config, model, features, df)


def explain_logistic_regression(
    config: dict,
    model: pyspark.ml.Model,
    model_features: List[str],
    df: pyspark.sql.DataFrame,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Computes the contribution of different features to the predicted output.

    Returns the contribution on both a 'macro' (group of features) and a 'micro'
    scale (individual features).

    Args:
        config: model configuration, as loaded by utils.get_config().
        model: the LogisticRegression model fit in the pipeline.
        model_features: the features used by the model.
        df: the prediction samples.

    Returns:
        A tuple consisting of:
        - a DataFrame with the contribution of groups of features (macro).
        - a DataFrame with the top 3 contributing features (micro).

    """
    # Explain prediction
    meso_features = [
        feature
        for features in config["FEATURE_GROUPS"].values()
        for feature in features
    ]

    # Get individual features contribution
    ep = (
        pyspark.ml.feature.ElementwiseProduct()
    )  # TODO Will need some adjustments because this line needs a SparkContext
    # initialized, assert?
    ep.setScalingVec(model.coefficients)
    ep.setInputCol("features")
    ep.setOutputCol("eprod")

    explanation_df = (
        ep.transform(df)
        .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
        .toDF(["siren"] + model_features)
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

    @F.udf(returnType=StringType())
    def get_max_value_colname(row, max_col):
        """Extract the column name where the nth highest value is found."""
        for i, name in enumerate(meso_features):
            if row[i] == max_col:
                return name
        raise NameError(f"Could not find columns associated to {max_col}")

    explanation_df = (
        explanation_df.withColumn(
            "1st_concerning_feat",
            get_max_value_colname(F.array(meso_features), "1st_concerning_val"),
        )
        .withColumn(
            "2nd_concerning_feat",
            get_max_value_colname(F.array(meso_features), "2nd_concerning_val"),
        )
        .withColumn(
            "3rd_concerning_feat",
            get_max_value_colname(F.array(meso_features), "3rd_concerning_val"),
        )
    )

    micro_scores_df = explanation_df.select(["siren"] + config["MICRO_SCORES_COLUMNS"])
    macro_scores_df = explanation_df.select(["siren"] + config["MACRO_SCORES_COLUMNS"])
    return macro_scores_df, micro_scores_df
