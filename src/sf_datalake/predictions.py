"""Post-processing of model predictions.

This module offers tools for:
- Merging multiple models outputs as a single prediction.
- Generating alert levels associated with scores.
- Tailoring alert levels based on "expert rules".

"""

import logging
from typing import Callable, Dict, List, Tuple

import pandas as pd


def merge_predictions(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    """Builds a list of predicted probabilities based on several model outputs.

    The available scores are picked by decreasing order of priority, that is, if a
    prediction is found for a given SIREN in the first model output, any subsequent
    prediction for this SIREN will be ignored.

    Args:
        predictions: A list of pandas DataFrame with the same set of columns, indexed by
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


def tailor_alert(
    predictions_df: pd.DataFrame,
    tailoring_steps: Dict[str, Tuple[Callable, Dict]],
    tailoring_rule: Callable[[pd.DataFrame], int],
    pre_tailoring_alert_col: str,
    post_tailoring_alert_col: str,
) -> pd.DataFrame:
    """Updates alert levels using expert rules.

    The `predictions_df` DataFrame should hold pre-computed data that can be fed, for
    each row (SIREN), to functions that will determine if tailoring conditions are met,
    which, in turn, will lead to a (potential) modification of alert levels.

    Args:
        predictions_df: Prediction data.
        tailoring_steps: A dict of {name: (function, **kwargs)} tuples of tailoring
          functions and their associated kwargs as a dict. Each function should take the
          predictions DataFrame as first argument and return a pd.Index that points rows
          where the corresponding tailoring condition is met.
        tailoring_rule: A mapping associating tailoring conditions to a post-tailoring
          alert level evolution (-1, 0 or 1). It should have a single argument in order
          to be called using `pd.df.apply` over a row's columns.
        pre_tailoring_alert_col: Name of the column holding before-tailoring alerts (as
          an int level).
        post_tailoring_alert_col: Name of the column in which to output after-tailoring
          alerts (as an int level).

    Returns:
        A prediction DataFrame with the `post_alert_col` containing a tailored alert
        level.

    """
    for name, (function, kwargs) in tailoring_steps.items():
        predictions_df[name] = False
        tailoring_index = function(**kwargs).intersection(predictions_df.index)
        predictions_df.loc[tailoring_index, name] = True

    predictions_df[post_tailoring_alert_col] = (
        predictions_df[pre_tailoring_alert_col]
        + predictions_df.apply(tailoring_rule, axis=1)
    ).clip(lower=0, upper=2)

    return predictions_df


def partial_unemployment_tailoring(
    pu_df: pd.DataFrame,
    pu_col: str,
    threshold: int,
) -> pd.Index:
    """Computes if alert level should be modified based on partial unemployment.

    The input DataFrame is indexed at the SIRET (entity) level, and this function helps
    determine if any of the company's entities has filed requests amounting to a maximal
    allowed value.

    Args:
        pu_df: Partial unemployment data, used to decide whether or not alert level
          should be updated. Index should be a list of SIRETs, not SIRENs.
        pu_col: Name of the column holding allowed partial unemployment duration.
        threshold: A number of months above which the tailoring switch is triggered.

    Returns:
        The (siren) indexes where an alert update should take place.

    """
    assert pu_df.index.name == "siret"
    pu_df["above_threshold"] = pu_df[pu_col] > threshold
    siren_mask = pu_df.groupby("siren")["above_threshold"].sum() > 0
    return siren_mask[siren_mask].index


def urssaf_debt_change(
    debt_df: pd.DataFrame,
    debt_cols: Dict[str, str],
    increasing: bool = True,
    tol: float = 0.2,
) -> pd.Index:
    """States if some debt value has increased/decreased.

    States if companies debt change over time, relative to the company's annual
    contributions, exceeds some input threshold.

    Args:
        debt_df: Debt data, used to decide whether or not alert level should be
          upgraded.
        debt_cols : A dict mapping names to lists of columns to be compared:
          - "start": The starting point of debt change.
          - "end": The ending point of debt change.
          - "contribution": An average annual contribution value.
        increasing: if `True`, points out cases where computed change is greater than
          `tol`. If `False`, points out cases where the opposite of computed change is
           greater than `tol`.
        tol: the threshold, as a percentage of debt / contributions change, above which
          the alert level should be updated.

    Returns:
        The (siren) indexes where an alert update should take place.

    """
    sign = 1 if increasing else -1
    debt_change = (
        sign
        * (debt_df[debt_cols["end"]] - debt_df[debt_cols["start"]])
        / (debt_df[debt_cols["contribution"]] * 12)
    )
    return debt_df[debt_change > tol].index
