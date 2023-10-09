#!/usr/bin/env python3
"""Generate a JSON document usable by the front-end application.

This script produces a JSON document that can be used by the front-end component of the
Signaux Faibles website to display info about selected companies. It requires data
output by the prediction model, as well as tailoring data:
- URSSAF debt / contributions data
- partial unemployment data
- financial data
- ...

See the command-line interface for more details on expected inputs.

"""

import argparse
import json
import sys
from typing import Union

import pandas as pd

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
    "--export_raw",
    action="store_true",
    help="Do not format as JSON. Only export raw data frame.",
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
path_group.add_argument(
    "--urssaf_tailoring_data",
    required=True,
    help="Path to a csv containing required urssaf tailoring data.",
)
path_group.add_argument(
    "--financial_tailoring_data",
    required=True,
    help="Path to a csv containing required financial tailoring data.",
)
path_group.add_argument(
    "--pu_tailoring_data",
    required=True,
    help="Path to a csv containing required partial unemployment tailoring data.",
)
parser.add_argument(
    "--pu_n_days",
    help="""Number of months to consider as upper threshold for partial unemployment
    tailoring.""",
    type=int,
    default=240,
)
parser.add_argument(
    "--concerning_threshold",
    default=None,
    type=float,
    help="""Threshold above which a `feature * weight` product is considered
    'concerning'.""",
)
parser.add_argument(
    "--algo_name",
    type=str,
    help="Name of the algorithm that produced the prediction",
    default=None,
)


def tailoring_rule(df_sig: pd.DataFrame) -> pd.Series:
    """Details how predictions should evolve based on tailorings.

    The tailoring rule will return the DataFrame with a supplementary int column
    having -1, 0, or 1 value that describes alert level evolution based on the truth
    values of some of the input's given elements.

    Args:
        df_sig: A DataFrame with tailoring signal columns.

    Returns:
        A delta alert column column holding either -1, 0, or 1, respectively
        corresponding to decrease, no change, increase of the alert level.

    """
    delta_alert_col = pd.Series(0, index=df_sig.index, dtype=int)
    n_financial_signals = (
        df_sig[
            [
                "solvabilité_faible",
                "k_propres_négatifs",
                "rentabilité_faible",
            ]
        ]
        .sum(axis=1)
        .astype(int)
    )
    delta_alert_col += n_financial_signals.map({0: 0, 1: 0, 2: 1, 3: 2})
    delta_alert_col += (
        df_sig["augmentation_dette_sur_cotisation_urssaf_recente"]
        & ~df_sig["dette_urssaf_macro_preponderante"]
    )
    delta_alert_col -= (
        df_sig["diminution_dette_urssaf_ancienne"]
        & df_sig["dette_urssaf_ancienne_significative"]
    )
    delta_alert_col += df_sig["demande_activite_partielle_elevee"]

    return delta_alert_col.clip(lower=-1, upper=1)


def normalize_siren(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIREN with zeroes if needed."""
    return x.astype(str).str.zfill(9)


def normalize_siret(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIRET with zeroes if needed."""
    return x.astype(str).str.zfill(13)


def identity(sig):
    """Dummy identity function."""
    return sig


# Parse CLI arguments, load predictions configuration and supplementary data

args = parser.parse_args()
configuration = sf_datalake.config.ConfigurationHelper(args.configuration)

micro_macro = {
    micro: macro
    for macro, micros in configuration.explanation.feature_groups.items()
    for micro in micros
}

algo_name = args.algo_name
if algo_name is None:
    algo_name = (
        "avec_paydex"
        if "retards_paiement" in configuration.explanation.feature_groups.keys()
        else "sans_paydex"
    )

additional_data = {
    "idListe": "Juin 2023",
    "batch": "2304",
    "algo": algo_name,
    "periode": "2023-06-01T00:00:00Z",
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
prediction_set["pre_tailoring_alert_group"] = prediction_set["probability"].apply(
    lambda x: 2 - (x < score_threshold[0.5]) - (x < score_threshold[2])
)

### A posteriori alert tailoring

# Partial unemployment
ap_df = pd.read_csv(
    args.pu_tailoring_data, index_col="ETAB_SIRET", dtype={"ETAB_SIREN": "str"}
)
ap_df.index = normalize_siret(ap_df.index)
ap_df["siren"] = ap_df.index.str[:9]
ap_df["n_jours"] = pd.to_timedelta(ap_df["n_jours"])
max_pu_days = ap_df.groupby("siren")["n_jours"].max()

# Urssaf tailoring
urssaf_df = pd.read_csv(
    args.urssaf_tailoring_data,
    dtype={"siren": str},
    parse_dates=["periode"],
).set_index("siren")
urssaf_df["dette"] = (
    urssaf_df["montant_part_patronale"] + urssaf_df["montant_part_ouvriere"]
)
assert hasattr(urssaf_df["periode"], "dt")

# Financial tailoring
financial_df = pd.read_csv(
    args.financial_tailoring_data,
    dtype={"siren": str},
).set_index("siren")

# Masks
old_debt_mask = urssaf_df["periode"].between(
    pd.Timestamp("2020-01-01"), pd.Timestamp("2021-09-01")
)
after_old_debt_mask = urssaf_df["periode"] >= pd.Timestamp("2021-09-01")
recent_period_start = pd.Timestamp("2022-09-01")
recent_period_end = pd.Timestamp("2022-12-01")
one_year_schedule_mask = urssaf_df["periode"].between(
    pd.Timestamp("2020-09-01"), pd.Timestamp("2021-08-31")
)

### Apply tailoring
tailoring_signals = {
    "solvabilité_faible": (identity, {"sig": financial_df["solvabilité_faible"]}),
    "k_propres_négatifs": (identity, {"sig": financial_df["k_propres_neg"]}),
    "rentabilité_faible": (identity, {"sig": financial_df["rentabilité_faible"]}),
    "diminution_dette_urssaf_ancienne": (
        sf_datalake.predictions.urssaf_debt_decrease_indicator,
        {
            "debt_p1": urssaf_df[old_debt_mask]["dette"],
            "debt_p2": urssaf_df[after_old_debt_mask]["dette"],
            "thresh": 0.1,
        },
    ),
    "dette_urssaf_ancienne_significative": (
        identity,
        {
            "sig": urssaf_df[one_year_schedule_mask]
            .groupby("siren")["dette_sur_cotisation_lissée"]
            .max()
            * 1
            / 12
            > 0.1
        },
    ),
    "augmentation_dette_sur_cotisation_urssaf_recente": (
        identity,
        {
            "sig": (
                urssaf_df[urssaf_df["periode"] == recent_period_end][
                    "dette_sur_cotisation_lissée"
                ]
                - urssaf_df[urssaf_df["periode"] == recent_period_start][
                    "dette_sur_cotisation_lissée"
                ]
            )
            * 1
            / 12
            > 0.1
        },
    ),
    "demande_activite_partielle_elevee": (
        sf_datalake.predictions.high_partial_unemployment_request_indicator,
        {
            "pu_s": max_pu_days,
            "threshold": pd.Timedelta(args.pu_n_days, unit="day"),
        },
    ),
    "dette_urssaf_macro_preponderante": (
        sf_datalake.predictions.urssaf_debt_prevails_indicator,
        {"macro_df": macro_explanation},
    ),
}

prediction_set = sf_datalake.predictions.compute_tailoring_signals(
    prediction_set,
    tailoring_signals,
)

## Export raw
if args.export_raw:
    prediction_set.to_csv(args.output_file)
    sys.exit()

# Fill missing values with False, we lose this information afterwards
prediction_set = prediction_set.fillna(
    value={col: 0 for col in tailoring_signals}
).astype({col: bool for col in tailoring_signals})
prediction_set = sf_datalake.predictions.tailor_alert(
    prediction_set,
    pre_tailoring_alert_col="pre_tailoring_alert_group",
    post_tailoring_alert_col="post_tailoring_alert_group",
    tailoring_function=tailoring_rule,
)


# Decode alert groups
alert_categories = pd.CategoricalDtype(
    categories=["Pas d'alerte", "Alerte seuil F2", "Alerte seuil F1"], ordered=True
)
prediction_set["alertPreRedressements"] = pd.Categorical.from_codes(
    codes=prediction_set["pre_tailoring_alert_group"], dtype=alert_categories
)
prediction_set["alert"] = pd.Categorical.from_codes(
    codes=prediction_set["post_tailoring_alert_group"], dtype=alert_categories
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
    list(tailoring_signals.keys())
    + ["pre_tailoring_alert_group", "post_tailoring_alert_group"],
    axis="columns",
).to_dict(orient="index")

for siren in prediction_set[
    prediction_set["alertPreRedressements"] != "Pas d'alerte"
].index:
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
for siren in prediction_set[prediction_set["alert"] != "Pas d'alerte"].index:
    output_entries[siren].update(
        {
            "redressements": [
                signal
                for signal in tailoring_signals
                if prediction_set.loc[siren, signal]
            ],
        },
    )


with open(args.output_file, mode="w", encoding="utf-8") as f:
    json.dump(
        [{"siren": siren, **props} for siren, props in output_entries.items()],
        f,
        indent=4,
    )
