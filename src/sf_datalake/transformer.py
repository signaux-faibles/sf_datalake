"""Transformer class.

Transformer is an abstract class to define different transformers.
"""
from typing import List

import pyspark.ml
import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import FloatType


def generate_stages(config: dict) -> List[Transformer]:
    """Generate all stages related to Transformers. Ready to be
    included in a pyspark.ml.Pipeline.

    Args:
        config : the config parameters (see sf_datalake.utils.get_config())

    Returns:
        List of prepared Transformers.
    """
    stages = []
    transformed_features = []
    for (features, transformer_name) in config["TRANSFORMERS"]:
        outputCol = f"features_to_transform_{transformer_name}"
        vector_assembler = VectorAssembler(
            inputCols=features,
            outputCol=outputCol,  # TODO is it necessary or overwritting
            # the same name during stages works?
        )
        transformer = get_transformer_from_str(transformer_name)
        stages += [vector_assembler, transformer]
        transformed_features += [outputCol]

    vector_assembler = VectorAssembler(
        inputCols=transformed_features, outputCol="features"
    )
    stages += [vector_assembler]
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


class FormatProbability(Transformer):  # pylint: disable=R0903
    # pylint disable to be consistent with the abstract class pyspark.ml.Transformer
    """A transformer to format the probability column in output of a model."""

    def _transform(self, dataset: pyspark.sql.DataFrame):  # pylint: disable=R0201
        """Extract the positive probability and cast it as float.

        Args:
            dataset: DataFrame to transform

        Returns:
            transformed DataFrame
        """
        transform_udf = F.udf(lambda v: float(v[1]), FloatType())
        return dataset.withColumn("probability", transform_udf("probability"))
