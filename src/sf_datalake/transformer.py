"""Transformer class.

Transformer is an abstract class to define different transformers.
"""
from typing import List

import pyspark.ml
from pyspark.ml.feature import StandardScaler, VectorAssembler


def generate_stages(config: dict) -> List[pyspark.ml.Transformer]:
    """Generate all stages related to Transformers. Ready to be
    included in a pyspark.ml.Pipeline.

    Args:
        config : the config parameters (see config.get_config())

    Returns:
        List of prepared Transformers.
    """
    stages = []
    transformed_features = []
    for (features, transformer_name) in config["TRANSFORMERS"]:
        output_col = f"features_to_transform_{transformer_name}"
        vector_assembler = VectorAssembler(
            inputCols=features,
            outputCol=output_col,  # TODO is it necessary or overwritting
            # the same name during stages works?
        )
        transformer = get_transformer_from_str(transformer_name)
        stages += [vector_assembler, transformer]
        transformed_features += [output_col]

    vector_assembler = VectorAssembler(
        inputCols=transformed_features, outputCol="features"
    )
    stages += [vector_assembler]
    return stages


def get_transformer_from_str(s: str) -> pyspark.ml.Transformer:
    """Get a Transformer from its name.

    Args:
        s: Name of the Transformer

    Returns:
        The selected Transformer with prepared parameters
    """
    factory = {
        "StandardScaler": StandardScaler(
            withMean=True,
            withStd=True,
            inputCol="features_to_transform_StandardScaler",
            outputCol="features_transformed_StandardScaler",
        )
    }
    return factory[s]
