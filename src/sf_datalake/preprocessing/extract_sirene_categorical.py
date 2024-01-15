"""Extract administrative data from sirene databases.

The input databases are:
- `StockUniteLegale`
- `StockEtablissement`
files.

See the argument parser help.

See https://www.data.gouv.fr/fr/datasets/
base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/
"""

import argparse

import pandas as pd

# Regions encoding

# pylint: disable=E1136
REGIONS: dict[str, str] = {
    "01": "Auvergne-Rhône-Alpes",
    "03": "Auvergne-Rhône-Alpes",
    "07": "Auvergne-Rhône-Alpes",
    "15": "Auvergne-Rhône-Alpes",
    "26": "Auvergne-Rhône-Alpes",
    "38": "Auvergne-Rhône-Alpes",
    "42": "Auvergne-Rhône-Alpes",
    "43": "Auvergne-Rhône-Alpes",
    "63": "Auvergne-Rhône-Alpes",
    "69": "Auvergne-Rhône-Alpes",
    "73": "Auvergne-Rhône-Alpes",
    "74": "Auvergne-Rhône-Alpes",
    "02": "Hauts-de-France",
    "59": "Hauts-de-France",
    "60": "Hauts-de-France",
    "62": "Hauts-de-France",
    "80": "Hauts-de-France",
    "04": "Provence-Alpes-Côte d'Azur",
    "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur",
    "13": "Provence-Alpes-Côte d'Azur",
    "83": "Provence-Alpes-Côte d'Azur",
    "84": "Provence-Alpes-Côte d'Azur",
    "08": "Grand Est",
    "10": "Grand Est",
    "51": "Grand Est",
    "52": "Grand Est",
    "54": "Grand Est",
    "55": "Grand Est",
    "57": "Grand Est",
    "67": "Grand Est",
    "68": "Grand Est",
    "88": "Grand Est",
    "09": "Occitanie",
    "11": "Occitanie",
    "12": "Occitanie",
    "30": "Occitanie",
    "31": "Occitanie",
    "32": "Occitanie",
    "34": "Occitanie",
    "46": "Occitanie",
    "48": "Occitanie",
    "65": "Occitanie",
    "66": "Occitanie",
    "81": "Occitanie",
    "82": "Occitanie",
    "14": "Normandie",
    "27": "Normandie",
    "50": "Normandie",
    "61": "Normandie",
    "76": "Normandie",
    "18": "Centre-Val de Loire",
    "28": "Centre-Val de Loire",
    "36": "Centre-Val de Loire",
    "37": "Centre-Val de Loire",
    "41": "Centre-Val de Loire",
    "45": "Centre-Val de Loire",
    "16": "Nouvelle-Aquitaine",
    "17": "Nouvelle-Aquitaine",
    "19": "Nouvelle-Aquitaine",
    "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine",
    "33": "Nouvelle-Aquitaine",
    "40": "Nouvelle-Aquitaine",
    "47": "Nouvelle-Aquitaine",
    "64": "Nouvelle-Aquitaine",
    "79": "Nouvelle-Aquitaine",
    "86": "Nouvelle-Aquitaine",
    "87": "Nouvelle-Aquitaine",
    "20": "Corse",
    "21": "Bourgogne-Franche-Comté",
    "25": "Bourgogne-Franche-Comté",
    "39": "Bourgogne-Franche-Comté",
    "58": "Bourgogne-Franche-Comté",
    "70": "Bourgogne-Franche-Comté",
    "71": "Bourgogne-Franche-Comté",
    "89": "Bourgogne-Franche-Comté",
    "90": "Bourgogne-Franche-Comté",
    "22": "Bretagne",
    "29": "Bretagne",
    "35": "Bretagne",
    "56": "Bretagne",
    "44": "Pays de la Loire",
    "49": "Pays de la Loire",
    "53": "Pays de la Loire",
    "72": "Pays de la Loire",
    "85": "Pays de la Loire",
    "75": "Île-de-France",
    "77": "Île-de-France",
    "78": "Île-de-France",
    "91": "Île-de-France",
    "92": "Île-de-France",
    "93": "Île-de-France",
    "94": "Île-de-France",
    "95": "Île-de-France",
    "97": "DROM",
    "98": "DROM",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
}

DROM: dict[str, str] = {
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "975": "Saint-Pierre-et-Miquelon",
    "976": "Mayotte",
    "977": "Saint-Barthélemy",
    "978": "Saint-Martin",
    "984": "Terres australes et antarctiques françaises",
    "986": "Wallis-et-Futuna",
    "987": "Polynésie française",
    "988": "Nouvelle-Calédonie",
    "989": "île Clipperton",
}

# Parse CLI arguments

parser = argparse.ArgumentParser("Extract sirene data")
parser.add_argument(
    "--ul_file", dest="UL_INPUT_FILE", help="The 'Unité légale' database."
)
parser.add_argument(
    "--et_file", dest="ET_INPUT_FILE", help="The 'Établissement' database."
)
parser.add_argument("-o", "--output_file", dest="OUTPUT_FILE")
args = parser.parse_args()

# "Établissement" data
df_et = (
    pd.read_csv(
        args.ET_INPUT_FILE,
        usecols=[
            "siren",
            "siret",
            "etablissementSiege",
            "codeCommuneEtablissement",
            "activitePrincipaleEtablissement",
        ],
        dtype={
            "siren": str,
            "siret": str,
            "codeCommuneEtablissement": str,
            "etablissementSiege": bool,
        },
    )
    .rename(
        columns={
            "etablissementSiege": "siège",
            "codeCommuneEtablissement": "code_commune",
            "activitePrincipaleEtablissement": "code_naf",
        },
    )
    .set_index("siren")
)

df_et["région"] = df_et["code_commune"].str[:2].map(REGIONS)
df_et.loc[df_et["région"] == "DROM", "région"] = (
    df_et[df_et["région"] == "DROM"]["code_commune"].str[:3].map(DROM)
)

# "Unité légale" data
df_ul = (
    pd.read_csv(
        args.UL_INPUT_FILE,
        usecols=["siren", "categorieJuridiqueUniteLegale"],
        dtype=str,
    )
    .set_index("siren")
    .rename(columns={"categorieJuridiqueUniteLegale": "catégorie_juridique"})
)

# Keep only head office
df_et = df_et.loc[df_et["siège"]].drop("siège", axis=1)

# Export
df_et.join(df_ul, on="siren", how="inner").to_csv(
    args.OUTPUT_FILE,
)
