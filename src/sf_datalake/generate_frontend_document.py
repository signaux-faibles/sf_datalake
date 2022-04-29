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
    if row["urssaf_debt_increase"]:
        return 1
    return 0


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
    def normalize_siren_index(ix: pd.Index, from_siret=False) -> pd.Index:
        if from_siret:
            return ix.astype(str).str.zfill(13).str[:9]
        return ix.astype(str).str.zfill(9)

    test_set = pd.read_csv(args["test_set"], index_col="siren")
    test_set.index = normalize_siren_index(test_set.index)

    prediction_set = pd.read_csv(args["prediction_set"], index_col="siren")
    prediction_set.index = normalize_siren_index(prediction_set.index)

    macro_explanation = pd.read_csv(args["explanation_data"], index_col="siren")
    macro_explanation.index = normalize_siren_index(macro_explanation.index)
    macro_explanation.columns = [
        col.replace("_macro_score", "") for col in macro_explanation.columns
    ]
    macro_explanation.drop(columns="misc", inplace=True, errors="ignore")

    concerning_data = pd.read_csv(args["concerning_data"], index_col="siren")
    concerning_data.index = normalize_siren_index(concerning_data.index)

    # Compute alert level thresholds
    score_threshold = sf_datalake.evaluation.optimal_beta_thresholds(
        predictions=test_set["probability"], outcomes=test_set["failure_within_18m"]
    )

    # Create encoded alert groups
    prediction_set["pre_tailoring_alert_group"] = prediction_set["probability"].apply(
        lambda x: 2 - (x < score_threshold[0.5]) - (x < score_threshold[2])
    )

    ### A posteriori alert tailoring
    # Urssaf tailoring
    urssaf_data = pd.read_csv(args["urssaf_data"], parse_dates=["periode"]).drop(
        "siren", axis=1
    )
    urssaf_data["siren"] = normalize_siren_index(urssaf_data.siret, from_siret=True)
    urssaf_data.set_index("siren", inplace=True)
    debt_start_data = urssaf_data[urssaf_data.periode == args["debt_start_date"]]
    debt_end_data = urssaf_data[urssaf_data.periode == args["debt_end_date"]]

    debt_start_agg = debt_start_data.groupby("siren").agg(
        {
            "cotisation": sum,
            "montant_part_ouvriere": sum,
            "montant_part_patronale": sum,
        }
    )
    debt_end_agg = debt_end_data.groupby("siren").agg(
        {
            "cotisation": sum,
            "montant_part_ouvriere": sum,
            "montant_part_patronale": sum,
        }
    )
    debt_data = debt_start_agg.join(debt_end_agg, lsuffix="_start", rsuffix="_end")
    debt_cols = {
        "start": ["montant_part_ouvriere_start", "montant_part_patronale_start"],
        "end": ["montant_part_ouvriere_end", "montant_part_patronale_end"],
        "contribution": ["cotisation_start", "cotisation_end"],
    }

    ### Apply tailoring
    tailoring_steps = [
        (
            "urssaf_debt_increase",
            sf_datalake.predictions.urssaf_debt_change,
            {
                "debt_df": debt_data,
                "debt_cols": debt_cols,
                "increasing": True,
                "tol": 0.2,
            },
        )
    ]
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
                    if output_entry[tailoring]
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
        "--urssaf_data",
        required=True,
        help="Path to a csv containing URSSAF debt data.",
    )
    parser.add_argument(
        "--start_date",
        dest="debt_start_date",
        type=str,
        default="2020-07-01",
        help="""The start date over which debt evolution will be computed
        (YYYY-MM-DD format).""",
    )
    parser.add_argument(
        "--end_date",
        dest="debt_end_date",
        type=str,
        default="2021-06-01",
        help="""The start date over which debt evolution will be computed
        (YYYY-MM-DD format).""",
    )
    parser.add_argument(
        "--concerning_threshold",
        default=None,
        type=float,
        help="""Threshold above which a `feature * weight` product is considered
        'concerning'.""",
    )

    main(parser.parse_args())
