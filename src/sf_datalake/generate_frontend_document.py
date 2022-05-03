"""Generate a JSON document usable by the front-end application.

This script produces a JSON document that can be used by the front-end component of the
Signaux Faibles website to display info about selected companies. It requires data
output by the prediction model, as well as tailoring data:
- URSSAF debt / contributions data
- partial unemployment data
- ...

See the command-line interface for more details on expected inputs.

"""

import argparse
import json
from typing import Union

import pandas as pd

import sf_datalake.evaluation
import sf_datalake.predictions
import sf_datalake.utils


def tailoring_rule(row) -> int:
    """Details how predictions should evolve based on tailorings.

    The tailoring rule will return an int (-1, 0, or 1) that describes alert level
    evolution based on the truth values of some of `row`'s given elements.

    Args:
        row: Any object that has a `__getitem__` method.

    Returns:
        Either -1, 0, or 1, respectively corresponding to decrease, no change, increase
        of the alert level.

    """
    if row["urssaf_new_debt_increase"]:
        return 1
    if row["urssaf_previous_debt_decrease"] and not row["urssaf_new_debt_increase"]:
        return -1
    return 0


def normalize_siren(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable SIREN with zeroes if needed."""
    return x.astype(str).str.zfill(9)


def normalize_siret(x: Union[pd.Series, pd.Index]) -> pd.Series:
    """Left pad an iterable of SIRET with zeroes if needed."""
    return x.astype(str).str.zfill(13).str[:9]


def main(
    args: Union[argparse.Namespace, dict],
):  # pylint: disable=too-many-locals
    """Generates a document that can be parsed for front-end integration.

    All arguments are described in the CLI argument parser help.

    Args:
        args: an argparse.Namespace or a dict containing all the arguments the arg
          parser is expecting.

    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    pred_config = sf_datalake.utils.get_config(args["configuration"])
    micro_macro = {
        micro: macro
        for macro, micros in pred_config["FEATURE_GROUPS"].items()
        for micro in micros
    }

    # Load prediction lists
    test_set = pd.read_csv(args["test_set"])
    test_set = test_set.set_index(normalize_siren(test_set["siren"]))

    prediction_set = pd.read_csv(args["prediction_set"])
    prediction_set = prediction_set.set_index(normalize_siren(prediction_set["siren"]))

    macro_explanation = pd.read_csv(args["explanation_data"])
    macro_explanation = macro_explanation.set_index(
        normalize_siren(macro_explanation["siren"])
    )
    macro_explanation.columns = [
        col.replace("_macro_score", "") for col in macro_explanation.columns
    ]
    macro_explanation.drop(columns="misc", inplace=True, errors="ignore")

    concerning_data = pd.read_csv(args["concerning_data"])
    concerning_data = concerning_data.set_index(
        normalize_siren(concerning_data["siren"])
    )

    # Compute alert level thresholds
    score_threshold = sf_datalake.evaluation.optimal_beta_thresholds(
        predictions=test_set["probability"], outcomes=test_set["failure_within_18m"]
    )

    # Create encoded alert groups
    prediction_set["pre_tailoring_alert_group"] = prediction_set["probability"].apply(
        lambda x: 2 - (x < score_threshold[0.5]) - (x < score_threshold[2])
    )

    ### A posteriori alert tailoring
    tailoring_data = pd.read_csv(args["tailoring_data"])
    tailoring_data["siret"] = normalize_siret(tailoring_data["siret"])
    tailoring_data["siren"] = tailoring_data["siret"].str[:9]

    # Urssaf tailoring
    debt_gb = tailoring_data.drop(["total_demande_ap", "siret"], axis=1).groupby(
        ["siren"]
    )
    debt_data = debt_gb.agg(sum)
    debt_data["dette_recente_reference"] = 0.0
    debt_data["dette_recente_courante"] = (
        debt_data["montant_part_patronale_recente_courante"]
        + debt_data["montant_part_ouvriere_recente_courante"]
    )
    debt_data["dette_ancienne_courante"] = (
        debt_data["montant_part_patronale_ancienne_courante"]
        + debt_data["montant_part_ouvriere_ancienne_courante"]
    )
    debt_data["dette_ancienne_reference"] = (
        debt_data["montant_part_patronale_ancienne_reference"]
        + debt_data["montant_part_ouvriere_ancienne_reference"]
    )
    new_debt_cols = {
        "start": "dette_recente_reference",
        "end": "dette_recente_courante",
        "contribution": "cotisation_moyenne_12m",
    }
    previous_debt_cols = {
        "start": "dette_ancienne_reference",
        "end": "dette_ancienne_courante",
        "contribution": "cotisation_moyenne_12m",
    }

    ### Apply tailoring
    tailoring_steps = {
        "urssaf_previous_debt_decrease": (
            sf_datalake.predictions.urssaf_debt_change,
            {
                "debt_df": debt_data,
                "debt_cols": previous_debt_cols,
                "increasing": False,
                "tol": 0.2,
            },
        ),
        "urssaf_new_debt_increase": (
            sf_datalake.predictions.urssaf_debt_change,
            {
                "debt_df": debt_data,
                "debt_cols": new_debt_cols,
                "increasing": True,
                "tol": 0.2,
            },
        ),
        "partial_unemployment_request": (
            sf_datalake.predictions.partial_unemployment_tailoring,
            {
                "pu_df": tailoring_data.set_index("siret"),
                "pu_col": "total_demande_ap",
                "threshold": args["n_months"],
            },
        ),
    }
    prediction_set = sf_datalake.predictions.tailor_alert(
        prediction_set,
        tailoring_steps,
        tailoring_rule,
        pre_tailoring_alert_col="pre_tailoring_alert_group",
        post_tailoring_alert_col="post_tailoring_alert_group",
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
    if args["metadata"] is not None:
        with open(args["metadata"], mode="r", encoding="utf-8") as md:
            for field, value in json.load(md).items():
                prediction_set[field] = value

    concerning_micro_threshold = args["concerning_threshold"]
    concerning_values_columns = [
        "1st_concerning_val",
        "2nd_concerning_val",
        "3rd_concerning_val",
    ]
    concerning_feats_columns = [
        "1st_concerning_feat",
        "2nd_concerning_feat",
        "3rd_concerning_feat",
    ]
    if concerning_micro_threshold is not None:
        mask = concerning_data[concerning_values_columns] > concerning_micro_threshold
        concerning_micro_variables = concerning_data[concerning_feats_columns].where(
            mask.values
        )
    else:
        concerning_micro_variables = concerning_data[concerning_feats_columns]

    ## Export
    alert_siren = prediction_set[prediction_set["alert"] != "Pas d'alerte"].index
    output_entry = prediction_set.to_dict(orient="index")
    for siren in alert_siren:
        output_entry[siren].update(
            {
                "macroRadar": macro_explanation.loc[siren].to_dict(),
                "redressements": [
                    tailoring
                    for tailoring in tailoring_steps
                    if output_entry[siren][tailoring]
                ],
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

    with open(args["output_file"], mode="w", encoding="utf-8") as f:
        json.dump(
            [{"siren": siren, **props} for siren, props in output_entry.items()],
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Generate a JSON document usable by the front-end Signaux Faibles
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
        help="Generated JSON document output path.",
    )
    path_group.add_argument(
        "--metadata",
        default=None,
        help="""Path to a JSON document containing additional metadata that will be
        added to every entry in the output document.""",
    )
    path_group.add_argument(
        "-c",
        "--configuration",
        help="Path to the prediction run configuration file.",
        required=True,
    )
    path_group.add_argument(
        "--concerning_data",
        required=True,
        help="""Path to a csv file containing data associated with the most 'concerning'
        features (i.e., the ones with highest values) values.""",
    )
    path_group.add_argument(
        "--tailoring_data",
        required=True,
        help="Path to a csv containing required tailoring data.",
    )
    path_group.add_argument(
        "--n_months",
        help="""Number of months to consider as upper threshold for partial unemployment
        tailoring.""",
        default=13,
    )
    parser.add_argument(
        "--concerning_threshold",
        default=None,
        type=float,
        help="""Threshold above which a `feature * weight` product is considered
        'concerning'.""",
    )

    main(parser.parse_args())
