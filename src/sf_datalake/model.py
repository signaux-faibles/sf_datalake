"""Model utilities and classes."""

from typing import List, Tuple

import pyspark.ml
import pyspark.ml.classification
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

import sf_datalake.utils


def generate_stages(config: dict) -> List[pyspark.ml.Model]:
    """Generate stages associated with a given Model.

    The returned list is ready to be included into a pyspark.ml.Pipeline object.

    Args:
        config: model configuration, as loaded by io.load_parameters().

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


def get_model_from_pipeline_model(
    pipeline_model: pyspark.ml.PipelineModel, model_name: str
) -> pyspark.ml.Model:
    """From a PipelineModel, extract the Model object based on its name.

    Args:
        pipeline_model : A PipelineModel representing a list of stages.
        model_name : The name of the model to be extracted.

    Raises:
        ValueError: If the model has not been found.

    Returns:
        The extracted model.

    """
    model = None
    for stage in pipeline_model.stages:
        if model_name in repr(stage):
            model = stage
            break

    if model is None:
        raise ValueError(
            f"Model with name {model_name} could not be found in pipeline stages."
        )

    return model


def explain(
    config: dict, pipeline_model: pyspark.ml.PipelineModel, df: pyspark.sql.DataFrame
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Computes the contribution of different features to the predicted output.

    May depend on used model type.

    Args:
        config: Model configuration, as loaded by io.load_parameters().
        pipeline_model: A fitted pipeline.
        df: The prediction samples.

    Returns:
        A tuple consisting of:
        - a DataFrame with the contribution of features groups (macro).
        - a DataFrame with the top 3 contributing features (micro).

    """
    model = get_model_from_pipeline_model(pipeline_model, config["MODEL"]["NAME"])
    factory = {"LogisticRegression": explain_logistic_regression}
    return factory[config["MODEL"]["NAME"]](config, model, df)


def explain_logistic_regression(
    config: dict,
    model: pyspark.ml.Model,
    df: pyspark.sql.DataFrame,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """Computes the contribution of different features to the predicted output.

    Returns the contribution on both a 'macro' (group of features) and a 'micro'
    scale (individual features).

    Args:
        config: model configuration, as loaded by io.load_parameters().
        model: the LogisticRegression model fit in the pipeline.
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
    ep = pyspark.ml.feature.ElementwiseProduct()
    ep.setScalingVec(model.coefficients)
    ep.setInputCol("features")
    ep.setOutputCol("eprod")

    explanation_df = (
        ep.transform(df)
        .rdd.map(lambda r: [r["siren"]] + [float(f) for f in r["eprod"]])
        .toDF(["siren"] + sf_datalake.utils.feature_index(config))
    )
    for group, features in config["MESO_GROUPS"].items():
        explanation_df = explanation_df.withColumn(
            group, sum(explanation_df[col] for col in features)
        ).drop(*features)

    # 'Macro' scores per group
    macro_scores_columns = [
        f"{group}_macro_score" for group in config["FEATURE_GROUPS"]
    ]
    for group, features in config["FEATURE_GROUPS"].items():
        explanation_df = explanation_df.withColumn(
            f"{group}_macro_score",
            1 / (1 + F.exp(-(sum(explanation_df[col] for col in features)))),
        )

    # 'Micro' scores
    micro_scores_columns = [
        "1st_concerning_val",
        "2nd_concerning_val",
        "3rd_concerning_val",
        "1st_concerning_feat",
        "2nd_concerning_feat",
        "3rd_concerning_feat",
    ]
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

    micro_scores_df = explanation_df.select(["siren"] + micro_scores_columns)
    macro_scores_df = explanation_df.select(["siren"] + macro_scores_columns)
    return macro_scores_df, micro_scores_df
