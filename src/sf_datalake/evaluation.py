"""Evaluation of model predictions. """

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimal_beta_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    betas: Iterable[float] = (0.5, 2.0),
    n_thr: int = 101,
) -> Dict[float, float]:
    """Computes the classification thresholds that maximise :math:`F_\\beta` score.

    We choose to define alert levels as the thresholds that maximize :math:`F_\\beta`
    for a given :math:`\\beta`. Typically, an alert threshold is tuned to favor
    precision (e.g., :math:`\\beta = 0.5`), while another alert threshold favors recall
    (e.g., :math:`\\beta = 0.5`).

    Args:
        y_true: The true outcomes. 0 means "no failure within the next 18
          months", while 1 means "failure within the next 18 months".
        y_score: The computed probability of a failure state within the next
          18 months.
        betas: The required :math:`\\beta` values for F-score thresholds
          computation.
        n_thr: Size of an even-spaced array of values spanning the [0, 1]
          interval that will be used as candidate threshold values.

    Returns:
        A dict of (beta, threshold) couples associated with each :math:`\\beta` input
          values.

    """
    thresh_array = np.linspace(0, 1, n_thr)

    f_beta = np.zeros((len(betas), n_thr))
    for n_t, threshold in enumerate(thresh_array):
        y_pred = y_score >= threshold
        for n_b, beta in enumerate(betas):
            f_beta[n_b, n_t] = fbeta_score(y_true, y_pred, beta)
    thresholds = thresh_array[np.argmax(f_beta, axis=1)]

    return dict(zip(betas, thresholds))


def metrics(
    y_true: np.array,
    y_score: np.array,
    beta: float = 1,
    thresh: float = 0.5,
) -> dict:
    """Computes multiple evaluation metrics for a binary classification model.

    Args:
        y_true: An array containing the true values.
        y_score: The computed probability of a failure state within the next
          18 months.
        beta: Weighting of recall relative to precision for the evaluation.
          Corresponds to the beta value of the F_beta score.
        thresh: The probability above which a binary model should classify a
          sample as positive.

    Returns:
        A dictionary containing the evaluation metrics.

    """
    y_pred = y_score >= thresh

    aucpr = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fbeta = fbeta_score(y_true, y_pred, beta=beta)

    return {
        "Confusion matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
        f"F{beta}-score": np.round(fbeta, 2),
        "Precision": np.round(precision, 2),
        "Recall": np.round(recall, 2),
        "Balanced accuracy": np.round(balanced_accuracy, 2),
        "Area under Precision-Recall curve": np.round(aucpr, 2),
        "Area under ROC curve": np.round(roc, 2),
    }
