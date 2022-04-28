"""Post-processing of model predictions.

This module offers tools for:
- Merging multiple models outputs as a single prediction.
- Generating alert levels associated with scores.
- Tailoring alert levels based on "expert rules".

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
    merged = predictions.pop(0)
    if not predictions:
        logging.warning("Predictions contains a single model output.")
        return merged
    assert all(merged.columns.equals(pred.columns) for pred in predictions)
    assert all(merged.index.name == pred.index.name for pred in predictions)

    for prediction in predictions:
        diff_ix = prediction.index.difference(merged.index)
        merged = merged.append(prediction.loc[diff_ix], verify_integrity=True)
    return merged


def name_alert_group(score: float, t_red: float, t_orange: float) -> str:
    """Returns an alert string associated with a score based on risk thresholds.

    This function can be called through `pd.DataFrame.apply` to create an alert column.

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
    predictions_df: pd.DataFrame,
    tailoring_steps: Iterable[Tuple[str, Callable, Dict]],
    tailoring_rule: Callable[pd.DataFrame, int],
    pre_alert_col: str,
    post_alert_col: str,
) -> pd.DataFrame:
    """Updates alert levels using expert rules.

    The `predictions_df` DataFrame should hold pre-computed data that can be fed, for
    each row (SIREN), to a function that outputs how an alert level will be modified
    (or not) by tailoring.

    Args:
        predictions_df: Prediction data.
        tailoring_steps: An iterable of (name, function, kwargs) tuples of tailoring
          functions and their associated kwargs as a dict. Each function should take the
          predictions DataFrame as first argument and return a pd.Index of rows where a
          tailoring condition is met.
        tailoring_rule: A mapping associating tailoring conditions to a post-tailoring
          alert level evolution (-1, 0 or 1). It should have a single argument in order
          to be called using `pd.df.apply`.
        pre_alert_col: Name of the column holding before-tailoring alerts.
        post_alert_col: Name of the column in which to output after-tailoring alerts.

    Returns:
        A prediction DataFrame with the `post_alert_col` containing a tailored alert
          level.

    """
    update_mapping = {
        ("Pas d'alerte", 1): "Alerte seuil F2",
        ("Pas d'alerte", 0): "Pas d'alerte",
        ("Pas d'alerte", -1): "Pas d'alerte",
        ("Alerte seuil F2", -1): "Pas d'alerte",
        ("Alerte seuil F2", 0): "Alerte seuil F2",
        ("Alerte seuil F2", 1): "Alerte seuil F1",
        ("Alerte seuil F1", -1): "Alerte seuil F2",
        ("Alerte seuil F1", 0): "Alerte seuil F1",
        ("Alerte seuil F1", 1): "Alerte seuil F1",
    }

    assert pre_alert_col in predictions_df.columns
    assert (
        predictions_df[pre_alert_col]
        .astype("category")
        .cat.categories.isin(update_mapping.values())
        .all()
    )

    for name, function, kwargs in tailoring_steps:
        predictions_df.loc[function(**kwargs), name] = True
        predictions_df.fillna({name: False}, inplace=True)

    predictions_df["tailoring_key"] = list(
        zip(predictions_df[pre_alert_col], predictions_df.apply(tailoring_rule, axis=1))
    )
    predictions_df[post_alert_col] = predictions_df["tailoring_key"].map(update_mapping)

    return predictions_df


def partial_unemployment_tailoring():
    """Computes if alert level should be modified based on partial unemployment."""


def urssaf_debt_tailoring(
    siren_index: pd.Index,
    debt_df: pd.DataFrame,
    debt_cols: Dict[str, List[str]],
    tol: float = 0.2,
) -> pd.Index:
    """Computes if alert level should be modified based on social debt.

    For a given company, if social debt evolution over time exceeds the input threshold,
    the corresponding index will be included in the rows for which alert should be
    updated.

    Args:
        siren_index: An index of the analyzed companies SIRENs.
        debt_df: Debt data, used to decide whether or not alert level should be
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
    debt_start = debt_df.loc[:, debt_cols["start"]].sum(axis="columns")
    debt_end = debt_df.loc[:, debt_cols["end"]].sum(axis="columns")
    contribution_average = (
        debt_df.loc[:, debt_cols["contribution"]]
        .mean(axis="columns")
        .replace(0, np.nan)
    )
    debt_evolution = ((debt_end - debt_start) / (contribution_average * 12)).fillna(0)
    return debt_df[debt_evolution > tol].index.intersection(siren_index)
