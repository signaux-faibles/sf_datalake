"""Transformer utilities and classes. """

from typing import Dict, List

import pyspark.ml
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import FloatType


def generate_stages(config: dict) -> List[Transformer]:
    """Generates all stages related to Transformer objects.

    The stages are ready to be included in a pyspark.ml.Pipeline.

    Args:
        config: model configuration, as loaded by utils.get_config().

    Returns:
        List of prepared Transformers.

    """
    stages: List[Transformer] = []
    transformed_features: List[str] = []
    transformer_features: Dict[str, List[str]] = {}
    for feature, transformer in config["TRANSFORMERS"]:
        transformer_features.setdefault(transformer, []).append(feature)
    for transformer, features in transformer_features.items():
        outputCol = f"features_to_transform_{transformer}"
        transformer_vector_assembler = VectorAssembler(
            inputCols=features, outputCol=outputCol
        )
        stages += [transformer_vector_assembler, get_transformer_from_str(transformer)]
        transformed_features.append(outputCol)

    concat_vector_assembler = VectorAssembler(
        inputCols=transformed_features, outputCol="features"
    )
    stages.append(concat_vector_assembler)
    return stages


def get_transformer_from_str(s: str) -> Transformer:
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


class ProbabilityFormatter(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to format the probability column in output of a model."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Extract the positive probability and cast it as float.

        Args:
            dataset: DataFrame to transform

        Returns:
            Transformed DataFrame with casted probability data.

        """
        transform_udf = F.udf(lambda v: float(v[1]), FloatType())
        return dataset.withColumn("probability", transform_udf("probability"))
