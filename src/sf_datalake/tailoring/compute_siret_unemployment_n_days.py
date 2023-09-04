"""Compute the number of requested unemployment days per SIRET.

There may be several overlapping requests for a given SIRET, this script will find the
largest contiguous periods of time and add their respective durations.

"""

#!/usr/bin/python3
# coding: utf-8
import argparse
from os import path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Compute number of unemployment days per SIRET"
)
parser.add_argument("INPUT_AP_FILE", help="Raw URSSAF data input file.")
parser.add_argument("OUTPUT_DIR")
parser.add_argument("--start_date", dest="START_DATE")
parser.add_argument("--end_date", dest="END_DATE")
args = parser.parse_args()

OUTPUT_AP_FILE = path.join(
    args.OUTPUT_DIR, f"n_days_{args.START_DATE}_to_{args.END_DATE}.csv"
)
AP_TAILORING_START_DATE = pd.Timestamp(args.START_DATE)
AP_TAILORING_END_DATE = pd.Timestamp(args.END_DATE)

df_ap = pd.read_csv(
    args.INPUT_AP_FILE,
    usecols=[
        "ETAB_SIRET",
        "DATE_DEB",
        "DATE_FIN",
        "MOTIF_RECOURS_SE",
        "EFF_AUTO",
        "EFF_ETAB",
    ],
)

df_ap = df_ap[df_ap["DATE_FIN"] < "2025"]  # clean spurious lines
df_ap["ETAB_SIRET"] = df_ap["ETAB_SIRET"].astype(str).str.zfill(14)

## Filters

# Type of request + size
df_ap = df_ap[
    (df_ap["MOTIF_RECOURS_SE"].astype(float) < 6.0)  # category
    & (df_ap["EFF_AUTO"] > df_ap["EFF_ETAB"] * 0.5)  # workforce ratio
]

# Filter according to period of interest
df_ap["DATE_DEB"] = pd.to_datetime(df_ap["DATE_DEB"])
df_ap["DATE_FIN"] = pd.to_datetime(df_ap["DATE_FIN"])
df_ap = df_ap[
    df_ap["DATE_DEB"].between(AP_TAILORING_START_DATE, AP_TAILORING_END_DATE)
    | df_ap["DATE_FIN"].between(AP_TAILORING_START_DATE, AP_TAILORING_END_DATE)
]
df_ap["DATE_DEB"].clip(lower=AP_TAILORING_START_DATE, inplace=True)
df_ap["DATE_FIN"].clip(upper=AP_TAILORING_END_DATE, inplace=True)

# Reindex
df_ap = df_ap.set_index(["ETAB_SIRET", "DATE_DEB"], drop=False).sort_index(
    level=["ETAB_SIRET", "DATE_DEB"]
)
# df_ap.sort_values(by="DATE_DEB", inplace=True)

# Sum number of requested days for each SIRET

n_jours = pd.Series(0, index=pd.Index(df_ap["ETAB_SIRET"].drop_duplicates()))
for siret in n_jours.index:
    df = df_ap.loc[(siret, slice(None)), ["DATE_DEB", "DATE_FIN"]].copy()
    df["group"] = (df["DATE_DEB"] > df["DATE_FIN"].shift().cummax()).cumsum()
    result = df.groupby("group").agg({"DATE_DEB": "min", "DATE_FIN": "max"})
    n_jours.loc[siret] = (result["DATE_FIN"] - result["DATE_DEB"]).sum(axis=0)

n_jours.name = "n_jours"
n_jours.to_csv(OUTPUT_AP_FILE, header=True)
