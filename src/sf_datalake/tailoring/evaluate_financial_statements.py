"""Evaluate financial statements for further prediction tailoring.

Expected inputs:
- Ratios file containing "dette_nette/caf", "ebe/ca", "ca", "k_propres",
  "annee_exercice" fields
- TVA file containing "d3310_01", "d3310_02", "dte_debut_periode",
  "dte_fin_periode" fields.
- A csv list of SIREN to be considered.

"""

#!/usr/bin/python3
# coding: utf-8
import argparse
from typing import Dict, Optional, Union

import pandas as pd

parser = argparse.ArgumentParser(
    description="Evaluate truth value of assertions on financial data."
)

parser.add_argument(
    "--ratios", help="Path to a csv of financial ratios.", dest="RATIOS_DATA"
)
parser.add_argument(
    "--tva",
    help="Path to a csv of tva data (d3310_01 and d3310_02) fields are expected",
    dest="TVA_DATA",
)
parser.add_argument(
    "--perimeter",
    help="Path to perimeter as a csv list of SIREN.",
    dest="SIREN_PERIMETER_DATA",
)
parser.add_argument(
    "--ratio_dette_sur_caf",
    dest="LIMITE_RATIO_DETTE_SUR_CAF",
    default=4.0,
)
parser.add_argument(
    "--lim_evolution_ratio_tva", dest="LIMITE_EVOLUTION_RATIO_TVA", default=0.8
)
parser.add_argument(
    "--cols_ca",
    dest="CA_COLS",
    default="d3310_01",
    choices=["d3310_01", "somme_champs"],
)

parser.add_argument("-o", "--output_file", dest="OUTPUT_FILE")
args = parser.parse_args()

siren_perimeter = (
    pd.read_csv(
        args.SIREN_PERIMETER_DATA,
        dtype={"siren": "str"},
    )
    .set_index("siren")
    .index
)

statements: Dict[str, pd.Index] = {}


def filter_n_most_recent(
    df: pd.DataFrame,
    date_col: str,
    min_date: str = "1970-01-01",
    max_date: str = "2100-01-01",
    n_last: int = 1,
):
    """Get the n most recent rows for every SIREN according to `date_col`

    The returned rows will be filtered inside a closed date interval defined by the
    `min_date` and `max_date` variables.

    Args:
        df: Input df.
        date_col: Name of the column containing date info.
        min_date: Minimum date (inclusive).
        max_date: Maximum date (inclusive).
        n_last: Number of most recent samples to extract.

    Returns:
        A filtered df.

    """
    time_mask = df[date_col].between(
        left=pd.Timestamp(min_date), right=pd.Timestamp(max_date)
    )
    df_filtered = df[time_mask].sort_values(by=date_col)
    return df_filtered.groupby(level="siren").tail(n_last)


def reindex_to_ref(
    statement: Union[pd.Series, pd.Series, pd.Index],
    ref_index: pd.Index,
    fill_value: Optional[bool] = None,
) -> pd.Series:
    """Align a pre-filtered object to another index, fill entries with boolean values.

    Args:
        statement: An indexed object. Its entries will be considered True wrt other
        input.
        ref_index: The index along which the input will be realigned.
        fill_value: A boolean value to fill the indexes that are not inside
        `statement`'s index.

    Returns:
        A boolean Series (possibly with some NaN values).

    """
    statement = pd.Series(True, index=statement.index)
    return statement.reindex(ref_index, fill_value=fill_value)


## Load ratios data
df_ratios = pd.read_csv(args.RATIOS_DATA, dtype={"siren": "str"}).set_index("siren")
df_ratios["annee_exercice"] = pd.to_datetime(df_ratios["annee_exercice"], format="%Y")
df_ratios = filter_n_most_recent(
    df_ratios, date_col="annee_exercice", min_date="2021-01-01", n_last=1
)

### Statements
ratios_siren_index = df_ratios.index.unique()

statements["solvabilité_faible"] = reindex_to_ref(
    df_ratios[df_ratios["dette_nette/caf"] > args.LIMITE_RATIO_DETTE_SUR_CAF],
    ratios_siren_index,
    fill_value=False,
).reindex(siren_perimeter, fill_value=None)

statements["ebe_neg"] = reindex_to_ref(
    df_ratios[(df_ratios["ebe/ca"] * df_ratios["ca"]) < 0],
    ratios_siren_index,
    fill_value=False,
).reindex(siren_perimeter, fill_value=None)

statements["k_propres_neg"] = reindex_to_ref(
    df_ratios[df_ratios["k_propres"] < 0], ratios_siren_index, fill_value=False
).reindex(siren_perimeter, fill_value=None)

## TVA (activité)

# - Quasiment aucune déclaration CA12 sur l'exercice 2022 (on attend décembre).
# - CA3 : 33k / 35k SIREN du périmètre ayant des déclarations

df_tva_ca3 = pd.read_csv(
    args.TVA_DATA,
    usecols=["siren", "dte_debut_periode", "dte_fin_periode", "d3310_01", "d3310_02"],
    parse_dates=["dte_debut_periode", "dte_fin_periode"],
    dtype={"siren": "str"},
).set_index("siren")

df_tva_ca3[["d3310_01", "d3310_02"]] = df_tva_ca3[["d3310_01", "d3310_02"]].fillna(0)

df_tva_ca3["durée_période"] = (
    df_tva_ca3["dte_fin_periode"]
    - df_tva_ca3["dte_debut_periode"]
    + pd.Timedelta(1, "D")
)


### Données post-covid (année 2022)
max_duration = (
    df_tva_ca3["dte_fin_periode"].max()
    - pd.Timestamp("2022-01-01")
    + pd.Timedelta(1, "D")
)

df_tva_ca3_postcovid = filter_n_most_recent(
    df_tva_ca3, min_date="2022-01-01", date_col="dte_debut_periode", n_last=12
)

postcovid_agg = df_tva_ca3_postcovid.groupby("siren").agg(
    {"dte_debut_periode": min, "durée_période": sum, "d3310_01": sum, "d3310_02": sum}
)
postcovid_agg = postcovid_agg[postcovid_agg["durée_période"] <= max_duration]

ca3_postcovid_durations = postcovid_agg["durée_période"].rename(
    "durée_déclarée_postcovid"
)


# ### Données pré-covid (année 2019)
df_tva_ca3_precovid = filter_n_most_recent(
    df_tva_ca3,
    min_date="2019-01-01",
    max_date="2020-01-01",
    date_col="dte_debut_periode",
    n_last=12,
)
min_debut_periode_precovid = (
    df_tva_ca3_precovid.groupby("siren")["dte_debut_periode"]
    .min()
    .rename("min_dte_debut_periode")
)
df_tva_ca3_precovid = df_tva_ca3_precovid.join(
    ca3_postcovid_durations, how="inner"
).join(min_debut_periode_precovid, how="inner")

df_tva_ca3_precovid = df_tva_ca3_precovid[
    df_tva_ca3_precovid["dte_fin_periode"]
    <= (
        df_tva_ca3_precovid["min_dte_debut_periode"]
        + df_tva_ca3_precovid["durée_déclarée_postcovid"]
    )
]

precovid_agg = df_tva_ca3_precovid.groupby("siren").agg(
    {"d3310_01": sum, "d3310_02": sum, "durée_période": sum}
)

df_tva_ca3_period_diff = (
    postcovid_agg.loc[precovid_agg.index, "durée_période"]
    - precovid_agg["durée_période"]
)

ca3_outlier_mask = (df_tva_ca3_period_diff > pd.Timedelta(-5, "D")) & (
    df_tva_ca3_period_diff < pd.Timedelta(5, "D")
)

precovid_agg["somme_champs"] = precovid_agg["d3310_01"] + precovid_agg["d3310_02"]
postcovid_agg["somme_champs"] = postcovid_agg["d3310_01"] + postcovid_agg["d3310_02"]

statements["activité_maintenue"] = (
    (
        postcovid_agg.reindex_like(precovid_agg).loc[ca3_outlier_mask, args.CA_COLS]
        >= args.LIMITE_EVOLUTION_RATIO_TVA
        * precovid_agg.loc[ca3_outlier_mask, args.CA_COLS]
    )
    .reindex(df_tva_ca3.index.unique(), fill_value=False)
    .reindex(siren_perimeter, fill_value=None)
)

statements["rentabilité_faible"] = (
    statements["activité_maintenue"] & statements["ebe_neg"]
)

pd.DataFrame.from_dict(statements).to_csv(args.OUTPUT_FILE)
