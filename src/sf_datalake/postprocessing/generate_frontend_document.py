#!/usr/bin/env python3
"""Generate a JSON document usable by the front-end application.

This script produces a JSON document that can be used by the front-end component of the
Signaux Faibles website to display info about selected companies. It requires data
output by the prediction model.

See the command-line interface for more details on expected inputs.

"""

import argparse
import datetime
import json
from os import path
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
    help="""Path to a directory containing csv files with "micro" and "macro"
    explanation data.""",
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

parser.add_argument(
    "--algo_name",
    type=str,
    help="Name of the algorithm that produced the prediction",
    default=None,
)


def normalize_siren(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIREN with zeroes if needed."""
    return x.astype(str).str.zfill(9)


def normalize_siret(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIRET with zeroes if needed."""
    return x.astype(str).str.zfill(13)


## Parse CLI arguments, load predictions configuration and supplementary data

args = parser.parse_args()
configuration = sf_datalake.configuration.ConfigurationHelper(args.configuration)

micro_macro = {
    micro: macro
    for macro, micros in configuration.explanation.topic_groups.items()
    for micro in micros
}

# pour avoir les mois en francais
mois = [
    "Janvier",
    "Février",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Août",
    "Septembre",
    "Octobre",
    "Novembre",
    "Décembre",
]
date = datetime.datetime.now()
imois = date.date().month
iyear = date.date().year

# additional_data = {
#    "idListe": "Mars 2024",
#    "batch": "2403",
#    "algo": importlib_metadata.version("sf_datalake"),
#    "période": "2024-03-01T00:00:00Z",
# }
additional_data = {
    "idListe": mois[imois - 1] + " " + str(iyear),
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

macro_explanation = pd.read_csv(
    path.join(args.explanation_data, "macro_explanation.csv")
)
macro_explanation["siren"] = normalize_siren(macro_explanation["siren"])
macro_explanation = macro_explanation.set_index("siren")
macro_explanation.columns = [
    col.replace("_macro_score", "") for col in macro_explanation.columns
]
# macro_explanation.drop(columns="misc", inplace=True, errors="ignore")


#############################################################################
# Convert macro_explanation for the waterfall :
# include expectation in a non-invasive way
# to keep interpretability of shap and by hinding expectation
proba = prediction_set["probability"]

# compute de sum of the macro expl
sum_macro = macro_explanation.iloc[:, :].sum(axis=1)

siren_index = macro_explanation.index.tolist()
for isi in siren_index:
    iproba = proba.loc[isi]
    iexp = iproba - sum_macro.loc[isi]
    ifactor = 100.0 * iproba / (iproba - iexp)
    macro_explanation.loc[isi] = ifactor * macro_explanation.loc[isi]

# rename quantities
macro_explanation = macro_explanation.rename(
    columns={"misc": "Variation de l'effectif de l'entreprise"}
)
macro_explanation = macro_explanation.rename(
    columns={"santé_financière": "Données financières"}
)
macro_explanation = macro_explanation.rename(
    columns={"activité_partielle": "Recours à l'activité partielle"}
)
macro_explanation = macro_explanation.rename(
    columns={"dette_urssaf": "Dettes sociales"}
)
macro_explanation = macro_explanation.rename(
    columns={"retards_paiement": "Retards de paiement fournisseurs"}
)

# End of rescaling part
#############################################################################


micro_explanation = pd.read_csv(
    path.join(args.explanation_data, "micro_explanation.csv")
)
micro_explanation["siren"] = normalize_siren(micro_explanation["siren"])
micro_explanation = micro_explanation.set_index("siren")

# Check for duplicated values
for name, df in {
    "Prediction": prediction_set,
    "Macro explanation": macro_explanation,
    "Micro explanation": micro_explanation,
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
    categories=["Pas d'alerte", "Alerte seuil F2", "Alerte seuil F1/2"], ordered=True
)
prediction_set["alert"] = pd.Categorical.from_codes(
    codes=prediction_set["alert_group"], dtype=alert_categories
)


# Convert probability to percentage
prediction_set["probability"] *= 100
prediction_set = prediction_set.rename(columns={"probability": "Risque de défaillance"})


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
            "macroExpl": macro_explanation.loc[siren].to_dict(),
            "microExpl": micro_explanation.loc[siren].to_dict(),
        }
    )

with open(args.output_file, mode="w", encoding="utf-8") as f:
    json.dump(
        [{"siren": siren, **props} for siren, props in output_entries.items()],
        f,
        indent=4,
    )
