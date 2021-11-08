"""Evaluation of model outputs.

"""

from typing import Tuple

import numpy as np
import pyspark
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def make_thresholds_from_fbeta(
    y_score: np.array,
    y_true: np.array,
    beta_F1: float = 0.5,
    beta_F2: float = 2,
    n_thr: int = 101,
) -> Tuple[float, float]:
    """Computes the classification thresholds that maximise :math:`F_\\beta` score.

    We choose to define both alert levels as the thresholds that maximize
    :math:`F_\\beta` for a given :math:`\\beta`. Typically, F1 alert threshold is tuned
    to favour precision (e.g., :math:`\\beta = 0.5`), while F2 alert threshold favors
    recall (e.g., :math:`\\beta = 0.5`).

    Args:
        y_score: The computed probability of a failure state within the next
          18 months.
        y_true: The predicted outcome. 0 means "no failure within the next 18
          months", while 1 means "failure within…".
        beta_F1: The :math:`\\beta` value for the first threshold.
        beta_F1: The :math:`\\beta` value for the second threshold.
        n_thr: an array of even-spaced `n_thr` values spanning the [0, 1] interval
          is used as evaluated threshold values.

    Returns:
        A couple of thresholds associated with the two input :math:`\\beta` values.

    """
    thresh = np.linspace(0, 1, n_thr)

    f_beta_F1 = []
    f_beta_F2 = []
    for thr in thresh:
        above_thresh = y_score >= thr
        F1_score = fbeta_score(y_true=y_true, y_pred=above_thresh, beta=beta_F1)
        F2_score = fbeta_score(y_true=y_true, y_pred=above_thresh, beta=beta_F2)
        f_beta_F1.append(F1_score)
        f_beta_F2.append(F2_score)

    t_F1 = thresh[np.argmax(f_beta_F1)]
    t_F2 = thresh[np.argmax(f_beta_F2)]

    return (t_F1, t_F2)


def metrics(
    y_true: np.array,
    y_score: np.array,
    beta: float = 1,
    thresh: float = 0.5,
) -> dict:
    """Computes multiple evaluation metrics for a binary classification model.

    Args:
        y_true: An array containing the true values.
        y_score: An array containing probabilities associated with each prediction.
        beta: Optional. If provided, weighting of recall relative to precision for the
          evaluation.
        thresh: Optional. If provided, the model will classify an entry X as positive
          if predict_proba(X)>=thresh. Otherwise, the model classifies X as positive if
          predict(X)=1, ie predict_proba(X)>=0.5

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


def print_spark_df_scores(results: pyspark.sql.DataFrame):
    """Quickly prints scores from data contained in a spark DataFrame."""
    correct_count = results.filter(results.label == results.prediction).count()
    total_count = results.count()
    correct_1_count = results.filter(
        (results.label == 1) & (results.prediction == 1)
    ).count()
    total_1_test = results.filter((results.label == 1)).count()
    total_1_predict = results.filter((results.prediction == 1)).count()

    print("All correct predections count: ", correct_count)
    print("Total count: ", total_count)
    print("Accuracy %: ", (correct_count / total_count) * 100)
    print("Recall %: ", (correct_1_count / total_1_test) * 100)
    print("Precision %: ", (correct_1_count / total_1_predict) * 100)
