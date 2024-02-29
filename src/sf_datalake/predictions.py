"""Post-processing of model predictions.

This module offers tools for:
- Merging multiple models outputs as a single prediction.

"""

import json
from typing import List


def merge_predictions_lists(predictions_paths: List[str], output_path: str):
    """Builds a front-end-ready predictions list based on multiple model outputs.

    The latest available information is used, that is, if a prediction is found for a
    given SIREN in any prediction list, it will replace any previous prediction for this
    same SIREN.

    Args:
        predictions: A list of paths to predictions JSON documents. Each entry in these
          documents should have at least a "siren" key.
        output_path: A path where the merged predictions list will be written.

    """
    predictions = []
    for path in predictions_paths:
        with open(path, encoding="utf-8") as f:
            predictions.append({entry["siren"]: entry for entry in json.load(f)})
    merged = predictions[0].copy()
    for prediction in predictions[1:]:
        merged.update(prediction)

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(
            list(merged.values()),
            f,
            indent=4,
        )
