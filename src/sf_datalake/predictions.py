"""Post-processing of model predictions.

This module offers tools for:
- Merging multiple models outputs as a single prediction.
- Generating alert levels associated with scores.
- Tailoring alert levels based on "expert rules".

"""

import json
from typing import Callable, Dict, List, Tuple

import pandas as pd


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


def high_partial_unemployment_request_indicator(
    pu_s: pd.Series,
    threshold: pd.Timedelta,
) -> pd.Index:
    """Computes an alert indicator based on partial unemployment request.

    The input DataFrame / Series simply gives .

    Args:
        pu_s: The number of requested partial unemployment days during a given timespan.
        threshold: A duration above which the tailoring switch is triggered.

    Returns:
        The (siren) indexes where partial unemployment requests level is deemed high.

    """
    return pu_s[pu_s > threshold].index


def urssaf_debt_decrease_indicator(
    debt_p1: pd.Series,
    debt_p2: pd.Series,
    thresh: float = 0.1,
) -> pd.Index:
    """States if some debt value has signficantly decreased.

    Debt is considered over two periods of time: `p1`, and `p2`. Each series should
    be indexed by siren and can hold multiple values for each given siren.

    Args:


    Returns:
        The (siren) indexes where debt change is (relatively) significant.

    """
    debt_p1_max = debt_p1.groupby("siren").max()
    debt_p2_min = debt_p2.groupby("siren").min()
    inter_index = debt_p1_max.index.intersection(debt_p2_min.index)
    mask = debt_p1_max[inter_index] > 0 & (
        debt_p2_min[inter_index] / debt_p1_max[inter_index] < thresh
    )
    return mask[mask].index


def urssaf_debt_vs_payment_schedule_indicator(
    debt_s: pd.Series,
    contribution_s: pd.Series,
    increasing: bool = True,
    thresh: float = 0.1,
) -> pd.Index:
    """States if some debt value has increased/decreased wrt to payment schedule.

    Returns True for indexes where companies debt, relative to monthly average
    contributions over some payment schedule, exceeds some input threshold.

    Args:
        debt_s: Debt data, used to decide whether or not alert level should be
          upgraded.
        debt_s :
        increasing: if `True`, points out cases where computed change is greater than
          `thresh`. If `False`, points out cases where the opposite of computed change
          is greater than `thresh`.
        thresh: The threshold, as a percentage of debt / contributions change, above
          which the alert level should be updated.

    Returns:
        The (siren) indexes where urssaf debt value is significant wrt contributions.

    """
    sign = 1 if increasing else -1
    debt_ratio = sign * debt_s / (contribution_s * 12)
    return debt_ratio[debt_ratio > thresh].index


def urssaf_debt_prevails_indicator(
    macro_df: pd.DataFrame,
) -> pd.Index:
    """States if URSSAF debt prevails among concerning predictors groups.

    Args:
        macro_df: Prediction macro-level influence for each variable category.

    Returns:
        The (siren) indexes where social debt prevails.

    """
    prevailing_mask = (macro_df.sub(macro_df["dette_urssaf"], axis=0) <= 0).all(axis=1)
    return macro_df[prevailing_mask].index
