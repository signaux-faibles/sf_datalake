"""Post-processing of model predictions.

This module offers tools for:
- Merging multiple models outputs as a single prediction.
- Generate alert levels associated with scores.
- Tailor alert levels based on "expert rules".

"""

import logging
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def merge_predictions(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    """Builds a list of predicted probabilities based on several model outputs.

    The available scores are picked by decreasing order of priority, that is, if a
    prediction is found for a given SIREN in the first model output, any subsequent
    prediction for this SIREN will be ignored.

    Args:
        model_list: A list of pandas DataFrame with the same set of columns, indexed by
          SIREN.

    Returns:
        A DataFrame with merged predicted probabilities.

    """
    merged = predictions.pop()
    if not predictions:
        logging.warning("Predictions contains a single model output.")
        return merged

    for prediction in predictions:
        diff_ix = prediction.index.difference(merged.index)
        merged = pd.append(
            prediction[diff_ix],
        )
    return merged


def name_alert_group(score: float, t_red: float, t_orange: float) -> str:
    """Returns an alert string associated with a score based on risk thresholds.

    Args:
        score: The input score.
        t_red: The highest risk threshold.
        t_orange: The lowest risk threshold.

    Returns:
        A string associated with the corresponding risk level.

    """
    if score >= t_red:
        return "Alerte seuil F1"
    if score >= t_orange:
        return "Alerte seuil F2"
    return "Pas d'alerte"


def tailor_alert(
    preds_df: pd.DataFrame,
    tailoring: Iterable[Tuple[Callable, Dict]],
    pre_alert_col: str = "alertPreRedressement",
    post_alert_col: str = "alert",
) -> pd.DataFrame:
    """Updates alert levels using expert rules.

    Args:
        preds_df: Prediction data.
        tailoring: An iterable of (function, kwargs) tuples of tailoring functions and
          their associated kwargs as a dict.
        pre_alert_col: Name of the alert column before tailoring.
        post_alert_col: Name of the alert column after tailoring.

    Returns:
        A prediction DataFrame with the `post_alert_col` containing a tailored alert
          level.

    """
    assert pre_alert_col in preds_df.columns

    update_rule = {
        "Pas d'alerte": "Alerte seuil F2",
        "Alerte seuil F2": "Alerte seuil F1",
        "Alerte seuil F1": "Alerte seuil F1",
    }

    preds_df.loc[:, post_alert_col] = preds_df.loc[:, pre_alert_col].copy()
    for function, kwargs in tailoring:
        update_index = function(**kwargs)
        updated_alert = preds_df.loc[:, post_alert_col].copy()
        preds_df.loc[update_index, post_alert_col] = updated_alert.loc[
            update_index
        ].map(update_rule)

    return preds_df


def debt_tailoring(
    siren_index: pd.DataFrame,
    debt_df: pd.DataFrame,
    debt_cols: Dict[str, List[str]],
    tol: float = 0.2,
) -> pd.Index:
    """Increases alert level based on social debt evolution over time.

    Args:
        siren_index: An index of the analyzed companies SIRENs.
        debt_df: Debt data, used as a means to decide whether alert level should be
          upgraded.
        debt_cols : A dict mapping names to lists of columns to be summed / averaged:
          - "start": The sum of these columns will be considered the starting point of
          debt change.
          - "end": The sum of these columns will be considered the ending point of debt
          change.
          - "contribution": These columns will be used to compute an average monthly
          contribution.
        tol: the threshold, as a percentage of (normalized) debt evolution, above which
          alert level is upgraded.

    Returns:
        The (siren) indexes where an alert update should take place.

    """

    # Compute debt change
    debt_start = debt_df.loc[:, debt_cols["start"]].sum(axis="columns")
    debt_end = debt_df.loc[:, debt_cols["end"]].sum(axis="columns")
    contribution_average = (
        debt_df.loc[:, debt_cols["contribution"]]
        .mean(axis="columns")
        .replace(0, np.nan, inplace=True)
    )
    debt_evolution = (debt_end - debt_start) / (contribution_average * 12).fillna(0)
    return debt_df[debt_evolution > tol].index.intersection(siren_index)
