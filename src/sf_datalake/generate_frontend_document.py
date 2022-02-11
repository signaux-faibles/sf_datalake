"""Generate a JSON document usable by the front-end application.

This script produces a JSON document that can be used by the front-end component of the
Signaux Faibles website to display info about selected companies. It requires data
output by the prediction model, as well as URSSAF debt / contributions data.

"""

import argparse
import json
from typing import Union

import pandas as pd

import sf_datalake.alert
import sf_datalake.evaluation


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

    pred_config = sf_datalake.utils.get_config(args.config)

    micro_macro = {
        micro: macro
        for macro, micros in pred_config["FEATURE_GROUPS"].items()
        for micro in micros
    }

    ## Load lists produced in the data lake.
    def normalize_siren_index(ix: pd.Index) -> pd.Index:
        return ix.astype(str).str.zfill(9)

    test_set = pd.read_csv(args["test_set"], header=0, index_col="siren")
    test_set.index = normalize_siren_index(test_set.index)
    test_set.rename({"probability": "score"}, axis=1, inplace=True)

    prediction_set = pd.read_csv(args["prediction_set"], header=0, index_col="siren")
    prediction_set.index = normalize_siren_index(prediction_set.index)
    prediction_set.rename({"probability": "score"}, axis=1, inplace=True)

    macro_explanation = pd.read_csv(
        args["explanation_data"], header=0, index_col="siren"
    )
    macro_explanation.index = normalize_siren_index(macro_explanation.index)
    macro_explanation.columns = [
        col[: col.find("_macro_score")] for col in macro_explanation.columns
    ]
    macro_explanation.drop(columns="misc", inplace=True, errors="ignore")

    concerning_data = pd.read_csv(args["concerning_data"], header=0, index_col="siren")
    concerning_data.index = normalize_siren_index(concerning_data.index)

    ## Compute thresholds
    score_threshold = sf_datalake.evaluation.optimal_beta_thresholds(
        predictions=test_set["score"], outcomes=test_set["label"]
    )
    prediction_set["alertPreRedressements"] = prediction_set["score"].apply(
        sf_datalake.alert.name_alert_group,
        args=(score_threshold["t1"], score_threshold["t2"]),
    )

    ## A posteriori alert tailoring
    urssaf_data = pd.read_csv(args["urssaf_data"])
    urssaf_data.index = normalize_siren_index(urssaf_data.index)
    debt_start_data = urssaf_data[urssaf_data.periode == args.debt_start_date]
    debt_end_data = urssaf_data[urssaf_data.periode == args.debt_end_date]

    debt_start_agg = debt_start_data.groupby("siren").agg(
        {"cotisation": sum, "part_salariale": sum, "part_patronale": sum}
    )
    debt_end_agg = debt_end_data.groupby("siren").agg(
        {"cotisation": sum, "part_salariale": sum, "part_patronale": sum}
    )
    debt_data = debt_start_agg.join(debt_end_agg, lsuffix="_start", rsuffix="_end")

    updated_alert, tailor_index = sf_datalake.alert.tailoring(
        prediction_set,
        debt_data,
        {
            "start": ["part_salariale_start", "part_patronale_start"],
            "end": ["part_salariale_end", "part_patronale_end"],
            "contribution": ["cotisation_start", "cotisation_end"],
        },
    )
    prediction_set.loc[:, "alert"] = updated_alert

    ## Score explanation per categories
    if args.metadata is not None:
        with open(args.metadata, mode="r", encoding="utf-8") as md:
            for field, value in json.load(md):
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
                "redressements": ["detteUrssaf"] if siren in tailor_index else [],
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
