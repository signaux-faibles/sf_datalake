"""Base configuration for learning algorithms.

TODO :
    - Should be refined when multiple models will be available.
    - Some hard-coded values should be available for user input.
"""

### User-set parameters

FILL_MISSING_VALUES = True
OVERSAMPLING_RATIO = 0.2
REGULARIZATION_COEFF = 0.05
N_CONCERNING_MICRO = 3
MAX_ITER = 50
TOL = 1e-5

TRAIN_DATES = ("2016-01-01", "2018-05-31")
TEST_DATES = ("2018-06-01", "2018-11-01")
PREDICTION_DATE = "2020-02-01"

### Variables definition

# CL2B recommended variables
MRV_VARIABLES = {
    "MNT_AF_BFONC_BFR",
    "MNT_AF_BFONC_TRESORERIE",
    "RTO_AF_RATIO_RENT_MBE",
    "MNT_AF_BFONC_FRNG",
    "MNT_AF_CA",
    "MNT_AF_SIG_EBE_RET",
    "RTO_AF_RENT_ECO",
    "RTO_AF_SOLIDITE_FINANCIERE",
    "RTO_INVEST_CA",
}

TAC_VARIABLES = {f"tac_1y_{v}" for v in MRV_VARIABLES}
# MRV_VARIABLES.update(TAC_VARIABLES)

PAYDEX_VARIABLES = {}
SUM_VARIABLES = {
    "cotisation",
    "cotisation_moy12m",
    "montant_part_ouvriere",
    "montant_part_ouvriere_past_1",
    "montant_part_ouvriere_past_12",
    "montant_part_ouvriere_past_2",
    "montant_part_ouvriere_past_3",
    "montant_part_ouvriere_past_6",
    "montant_part_patronale",
    "montant_part_patronale_past_1",
    "montant_part_patronale_past_12",
    "montant_part_patronale_past_2",
    "montant_part_patronale_past_3",
    "montant_part_patronale_past_6",
    "effectif",
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
}

AVG_VARIABLES = {
    "ratio_dette_moy12m",
}

COMP_VARIABLES = {
    "ratio_dette",
    "avg_delta_dette_par_effectif",
}

SF_VARIABLES = SUM_VARIABLES | AVG_VARIABLES | COMP_VARIABLES

BASE_VARIABLES = {
    "periode",
    "siren",
    "code_naf",
    "time_til_failure",
}
TARGET_VARIABLE = {"failure_within_18m"}
FEATURES = SF_VARIABLES | MRV_VARIABLES

STD_SCALE_FEATURES = list(FEATURES)

SF_DEFAULT_VALUES = {
    "time_til_failure": 9999,
    ### ACOSS
    "montant_part_ouvriere_past_12": 0.0,
    "montant_part_patronale_past_12": 0.0,
    "montant_part_ouvriere_past_6": 0.0,
    "montant_part_patronale_past_6": 0.0,
    "montant_part_ouvriere_past_3": 0.0,
    "montant_part_patronale_past_3": 0.0,
    "montant_part_ouvriere_past_2": 0.0,
    "montant_part_patronale_past_2": 0.0,
    "montant_part_ouvriere_past_1": 0.0,
    "montant_part_patronale_past_1": 0.0,
    "cotisation": 0.0,
    "montant_part_ouvriere": 0.0,
    "montant_part_patronale": 0.0,
    "cotisation_moy12m": 0.0,
    "ratio_dette": 0.0,
    "ratio_dette_moy12m": 0.0,
    ### Activité partielle
    "apart_heures_autorisees": 0.0,
    "apart_heures_consommees_cumulees": 0.0,
    "apart_heures_consommees": 0.0,
    "avg_delta_dette_par_effectif": 0.0,
    ### Effectif
    "effectif": 0,
    "effectif_ent": 0,
}

MESO_URSSAF_GROUPS = {
    "cotisation_urssaf": ["cotisation", "cotisation_moy12m"],
    "part_ouvriere": [
        "montant_part_ouvriere",
        "montant_part_ouvriere_past_1",
        "montant_part_ouvriere_past_2",
        "montant_part_ouvriere_past_3",
        "montant_part_ouvriere_past_6",
        "montant_part_ouvriere_past_12",
    ],
    "part_patronale": [
        "montant_part_patronale",
        "montant_part_patronale_past_1",
        "montant_part_patronale_past_2",
        "montant_part_patronale_past_3",
        "montant_part_patronale_past_6",
        "montant_part_patronale_past_12",
    ],
    "dette": [
        "ratio_dette",
        "ratio_dette_moy12m",
        "avg_delta_dette_par_effectif",
    ],
}

FEATURE_GROUPS = {
    "sante_financiere": [
        "MNT_AF_BFONC_BFR",
        "MNT_AF_BFONC_FRNG",
        "MNT_AF_BFONC_TRESORERIE",
        "MNT_AF_CA",
        "MNT_AF_SIG_EBE_RET",
        "RTO_AF_RATIO_RENT_MBE",
        "RTO_AF_RENT_ECO",
        "RTO_AF_SOLIDITE_FINANCIERE",
        "RTO_INVEST_CA",
    ],
    "activite_partielle": [
        "apart_heures_consommees",
        "apart_heures_consommees_cumulees",
    ],
    "dette_urssaf": [
        "cotisation_urssaf",
        "part_ouvriere",
        "part_patronale",
        "dette",
    ],
    "misc": ["effectif"],
    "retards_paiement": ["paydex_group", "paydex_yoy"],
}
