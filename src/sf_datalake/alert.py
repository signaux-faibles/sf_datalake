"""Generation of alert levels based on model predictions. """

from typing import Dict, List

import numpy as np
import pandas as pd


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


def alert_tailoring(
    preds_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    debt_cols: Dict[str, List[str]],
    prev_alert_col: str = "alertPreRedressements",
    tol: float = 0.2,
):
    """Expert rule for alert update.

    Args:
        preds_df: Prediction data, containing an alert column.
        debt_df: Debt data, used as a means to decide whether alert level should be
          upgraded.
        debt_cols : A dict mapping names to lists of columns to be summed / averaged:
          - "start": The sum of these columns will be considered the starting point of
            debt change.
          - end: The sum of these columns will be considered the ending point of debt
            change.
          - contribution: These columns will be used to compute an average monthly
            contribution.
        prev_alert_col: the name of the initial alert level column in `preds_df`.
        tol: the threshold, as a percentage of (normalized) debt evolution, above which
          alert level is upgraded.

    Returns:
        A column of updated alerts as well as the corresponding indexes where an update
          took place.

    """
    update_rule = {
        "Pas d'alerte": "Alerte seuil F2",
        "Alerte seuil F2": "Alerte seuil F1",
        "Alerte seuil F1": "Alerte seuil F1",
    }

    # Compute debt change
    debt_start = debt_df[debt_cols["start"]].sum(axis="columns")
    debt_end = debt_df[debt_cols["end"]].sum(axis="columns")
    contribution_average = (
        debt_df[debt_cols["contribution"]]
        .mean(axis="columns")
        .replace(0, np.nan, inplace=True)
    )
    debt_evolution = (debt_end - debt_start) / (contribution_average * 12).fillna(0)

    update_index = debt_df[debt_evolution > tol].index.intersection(preds_df.index)
    updated_alert = preds_df[prev_alert_col].copy()
    updated_alert.loc[update_index] = updated_alert[update_index].map(update_rule)
    return updated_alert, update_index
