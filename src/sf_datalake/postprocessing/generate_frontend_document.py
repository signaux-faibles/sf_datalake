#!/usr/bin/env python3
"""Generate a JSON document usable by the front-end application.

This script produces a JSON document that can be used by the front-end component of the
Signaux Faibles website to display info about selected companies. It requires data
output by the prediction model.

See the command-line interface for more details on expected inputs.

"""

import argparse
import json
from typing import Union

import importlib_metadata
import pandas as pd

import sf_datalake.configuration
import sf_datalake.evaluation
import sf_datalake.io
import sf_datalake.predictions
import sf_datalake.utils

parser = argparse.ArgumentParser(
    description="""Generate a JSON document to be fed to the Signaux Faibles front-end
    application."""
)

path_group = parser.add_argument_group(
    "paths", description="Path command line arguments."
)

path_group.add_argument(
    "-t", "--test_set", required=True, help="Path to the test set csv file."
)
path_group.add_argument(
    "-p",
    "--prediction_set",
    required=True,
    help="Path to the prediction set csv file.",
)
path_group.add_argument(
    "-x",
    "--explanation_data",
    required=True,
    help="""Path to a directory containing csv files with partial categorized
    'explanation' scores.""",
)
path_group.add_argument(
    "-o",
    "--output_file",
    required=True,
    help="Generated output path.",
)
path_group.add_argument(
    "--configuration",
    help="Path to the configuration file.",
    required=True,
)
path_group.add_argument(
    "--concerning_data",
    required=True,
    help="""Path to a csv file containing data associated with the most 'concerning'
    features (i.e., the ones with highest values) values.""",
)
parser.add_argument(
    "--concerning_threshold",
    default=None,
    type=float,
    help="""Threshold above which a `feature * weight` product is considered
    'concerning'.""",
)


def normalize_siren(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIREN with zeroes if needed."""
    return x.astype(str).str.zfill(9)


def normalize_siret(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIRET with zeroes if needed."""
    return x.astype(str).str.zfill(13)


# Parse CLI arguments, load predictions configuration and supplementary data

args = parser.parse_args()
configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)

micro_macro = {
    micro: macro
    for macro, micros in configuration.explanation.topic_groups.items()
    for micro in micros
}

additional_data = {
    "idListe": "Mars 2024",
    "batch": "2403",
    "algo": importlib_metadata.version("sf_datalake"),
    "période": "2024-03-01T00:00:00Z",
}

# Load prediction lists
test_set = pd.read_csv(args.test_set)
test_set["siren"] = normalize_siren(test_set["siren"])
test_set = test_set.set_index("siren")

prediction_set = pd.read_csv(args.prediction_set)
prediction_set["siren"] = normalize_siren(prediction_set["siren"])
prediction_set = prediction_set.set_index("siren")

macro_explanation = pd.read_csv(args.explanation_data)
macro_explanation["siren"] = normalize_siren(macro_explanation["siren"])
macro_explanation = macro_explanation.set_index("siren")
macro_explanation.columns = [
    col.replace("_macro_score", "") for col in macro_explanation.columns
]
macro_explanation.drop(columns="misc", inplace=True, errors="ignore")

concerning_data = pd.read_csv(args.concerning_data)
concerning_data["siren"] = normalize_siren(concerning_data["siren"])
concerning_data = concerning_data.set_index("siren")

# Check for duplicated values
for name, df in {
    "prediction": prediction_set,
    "macro radar": macro_explanation,
    "concerning values": concerning_data,
}.items():
    if df.index.duplicated().any():
        raise ValueError(
            f"{name} dataframe has duplicated index values: \n \
            {df.index[df.index.duplicated()]}"
        )

# Compute alert level thresholds
score_threshold = sf_datalake.evaluation.optimal_beta_thresholds(
    y_true=test_set["failure"], y_score=test_set["probability"]
)

# Create encoded alert groups
prediction_set["alert_group"] = prediction_set["probability"].apply(
    lambda x: 2 - (x < score_threshold[0.5]) - (x < score_threshold[2])
)

# Decode alert groups
alert_categories = pd.CategoricalDtype(
    categories=["Pas d'alerte", "Alerte seuil F2", "Alerte seuil F1"], ordered=True
)
prediction_set["alert"] = pd.Categorical.from_codes(
    codes=prediction_set["alert_group"], dtype=alert_categories
)

## Score explanation per categories
n_concerning_micro = configuration.explanation.n_concerning_micro
concerning_micro_threshold = args.concerning_threshold
concerning_values_columns = [f"concerning_val_{n}" for n in range(n_concerning_micro)]
concerning_feats_columns = [f"concerning_feat_{n}" for n in range(n_concerning_micro)]
if concerning_micro_threshold is not None:
    mask = concerning_data[concerning_values_columns] > concerning_micro_threshold
    concerning_micro_variables = concerning_data[concerning_feats_columns].where(
        mask.values
    )
else:
    concerning_micro_variables = concerning_data[concerning_feats_columns]

## Export front json document
for field, value in additional_data.items():
    prediction_set[field] = value

output_entries = prediction_set.drop(
    ["alert_group"],
    axis="columns",
).to_dict(orient="index")

for siren in prediction_set[prediction_set["alert"] != "Pas d'alerte"].index:
    output_entries[siren].update(
        {
            "macroRadar": macro_explanation.loc[siren].to_dict(),
            "explSelection": {
                "selectConcerning": [
                    [micro_macro[micro], micro]
                    for micro in filter(
                        pd.notna, concerning_micro_variables.loc[siren].values
                    )
                ]
            },
        }
    )


with open(args.output_file, mode="w", encoding="utf-8") as f:
    json.dump(
        [{"siren": siren, **props} for siren, props in output_entries.items()],
        f,
        indent=4,
    )
