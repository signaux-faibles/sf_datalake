"""Compute a set of evaluation metrics on predictions.
"""

import os

import pandas as pd

import sf_datalake.evaluation

MODEL_OUTPUT_DIR = ""
PREDICTION_LIST_ID = ""
DROP_STRONG_SIGNALS = False
WITH_PAYDEX = False

test_file = os.path.join(
    MODEL_OUTPUT_DIR,
    PREDICTION_LIST_ID,
    "paydex" if WITH_PAYDEX else "standard",
    "test_data.csv",
)
test_data = pd.read_csv(
    test_file,
    dtype={
        "failure": bool,
        "siren": str,
        "probability": float,
    },
    index_col=0,
)

if DROP_STRONG_SIGNALS:
    test_data = test_data[test_data["time_til_failure"] > 0]

thresholds = sf_datalake.evaluation.optimal_beta_thresholds(
    y_true=test_data["failure"],
    y_score=test_data["probability"],
)

red_evaluation = sf_datalake.evaluation.metrics(
    y_true=test_data["failure"],
    y_score=test_data["probability"],
    beta=0.5,
    thresh=thresholds[0.5],
)
orange_evaluation = sf_datalake.evaluation.metrics(
    y_true=test_data["failure"],
    y_score=test_data["probability"],
    beta=2,
    thresh=thresholds[2],
)

print(
    f"Producing metrics over {PREDICTION_LIST_ID} list "
    f"{'paydex' if WITH_PAYDEX else 'standard'} test set."
)
print(f"Strong signals are {'' if DROP_STRONG_SIGNALS else 'NOT '}dropped")
print(f"Number of SIREN in test dataset: {len(test_data)}")
print("Red level")
print(red_evaluation)
print("Orange level")
print(orange_evaluation)
