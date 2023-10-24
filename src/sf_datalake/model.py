"""Model utilities and classes."""


import pyspark.ml


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
