"""Model utilities and classes."""


import pyspark.ml
import pyspark.ml.classification


def get_model_from_conf(model_config: dict, target_col: str) -> pyspark.ml.Model:
    """Generates a Model object from a given configuration.

    Args:
        model_config: The Model configuration. The dict contains parameters that
          corresponds to some pyspark.ml.Model arguments.
        target_col: The target column's name.

    Returns:
        The selected Model instantiated using the input config parameters.

    """
    factory = {
        "LogisticRegression": pyspark.ml.classification.LogisticRegression(
            labelCol=target_col,
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
