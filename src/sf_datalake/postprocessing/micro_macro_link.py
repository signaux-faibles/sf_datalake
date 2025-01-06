#!/usr/bin/env python3
"""Make the link between macro/micro explanation and rescale the micro for the waterfall

This script build the link between macro and micro quantities.
It then rescale micro quantities to recover the corresponding
 macro quantity just by using a summation.

"""

import numpy as np


def scaleMicro(macro, micro):
    """
    Compute scaled micro and build
     dictionary output
    """
    values = np.array(list(micro.values()))
    factor = macro / np.sum(values)
    values *= factor
    dic_ = dict(zip(list(micro.keys()), values))
    return dic_


def getRescaledData(macro, micro, micro_macro):
    """
    For each macro quantity,
    rescale corresponding micro quantities
    """

    output = {}
    for nmacro, imacro in macro.items():
        list_micro_names = []
        for nmm, imm in micro_macro.items():
            if imm == nmacro:
                list_micro_names.append(nmm)
        filtered_micro = {
            key: value
            for key, value in micro.items()
            if any(key.startswith(prefix) for prefix in list_micro_names)
        }
        dic_ = scaleMicro(imacro, filtered_micro)
        output[nmacro] = dic_
    return output
