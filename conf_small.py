VARIABLES = [
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
    "ratio_dette",
    "ratio_dette_moy12m",
    "effectif",
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
]

# ces variables sont toujours requêtées
VARIABLES += ["outcome", "periode", "siret", "siren", "time_til_outcome", "code_naf"]

FEATURES = [
    "apart_heures_consommees_cumulees",
    "apart_heures_consommees",
    "ratio_dette",
    "avg_delta_dette_par_effectif",
]

TO_ONEHOT_ENCODE = []
TO_SCALE = list(set(FEATURES) - set(TO_ONEHOT_ENCODE))
