#!/usr/bin/env python3
"""Make the link between macro/micro explanation and rescale the micro for the waterfall

This script build the link between macro and micro quantities.
It then rescale micro quantities to recover the corresponding
 macro quantity just by using a summation.

"""

import numpy as np


## To make the link macro/micro
def getMacroFromMicro(toCheck):
    """
    For each macro quantity, it will extract
     the corresponding micro names
    """
    dette_urssaf = [
        "cotisation",
        "dette_sociale_ouv",
        "dette_sociale_pat",
        "dette_sur_cotisation_m",
        "dette_par_effectif",
    ]
    activite_partielle = ["ap_heures_conso"]
    retards_paiement = ["paydex", "fpi_30", "fpi_90"]

    respective_macro = ""
    continueMacro = True

    if toCheck.startswith("effectif"):
        respective_macro = "Variation de l'effectif de l'entreprise"
        continueMacro = False
    if continueMacro:
        for idu in dette_urssaf:
            if idu in toCheck:
                respective_macro = "Dettes sociales"
                continueMacro = False
    if continueMacro:
        for iap in activite_partielle:
            if iap in toCheck:
                respective_macro = "Recours à l'activité partielle"
                continueMacro = False
    if continueMacro:
        for rp in retards_paiement:
            if rp in toCheck:
                respective_macro = "Retards de paiement fournisseurs"
                continueMacro = False
    if continueMacro:
        respective_macro = "Données financières"
    return respective_macro


## Extract micro values and name
def getMicroOfMacro(micro_to_macro, nom_micro, micro, nomMacro):
    """
    For each micro of a given macro it
    will extract micro values and names
    """

    pos_ = np.argwhere(micro_to_macro == nomMacro)[:, 0]
    nom_micro_inMacro = []
    for ip in pos_:
        nom_micro_inMacro.append(nom_micro[ip])
    micro_values = np.array([micro[nm] for nm in nom_micro_inMacro])
    return micro_values, nom_micro_inMacro


def scaleMicro(macro, micro_values, micro_names):
    """
    Compute scaled micro and build
     dictionary output
    """
    factor = macro / np.sum(micro_values)
    micro_values *= factor
    dic_ = dict(zip(micro_names, micro_values))
    return dic_


## All the process for 1 macro
def rescaleMicroOfMacro(macro, micro_to_macro, nom_micro, micro, name):
    """ "
    All the process to rescale
    micro of given macro
    and return dic
    """
    micro_, micro_name = getMicroOfMacro(micro_to_macro, nom_micro, micro, name)
    dic_ = scaleMicro(macro[name], micro_, micro_name)
    return dic_


# Rescale values
def getRescaledData(macro, micro):
    """
    For each macro quantity,
    rescale corresponding micro quantities
    """

    # Nom des quantites micro
    nom_micro = list(micro.keys())

    # Pour chaque quantite micro on lui affecte la quantite macro associee
    micro_to_macro = []
    for inm in nom_micro:
        i_macro = getMacroFromMicro(inm)
        micro_to_macro.append(i_macro)
    micro_to_macro = np.array(micro_to_macro)

    # Pour chaque quantite macro on recupere les valeurs micro et leurs noms
    dic_di = rescaleMicroOfMacro(
        macro,
        micro_to_macro,
        nom_micro,
        micro,
        "Variation de l'effectif de l'entreprise",
    )
    dic_sf = rescaleMicroOfMacro(
        macro, micro_to_macro, nom_micro, micro, "Données financières"
    )
    dic_ap = rescaleMicroOfMacro(
        macro, micro_to_macro, nom_micro, micro, "Recours à l'activité partielle"
    )
    dic_du = rescaleMicroOfMacro(
        macro, micro_to_macro, nom_micro, micro, "Dettes sociales"
    )
    dic_rp = rescaleMicroOfMacro(
        macro, micro_to_macro, nom_micro, micro, "Retards de paiement fournisseurs"
    )

    output = {}
    output["Variation de l'effectif de l'entreprise"] = dic_di
    output["Données financières"] = dic_sf
    output["Recours à l'activité partielle"] = dic_ap
    output["Dettes sociales"] = dic_du
    output["Retards de paiement fournisseurs"] = dic_rp

    return output
