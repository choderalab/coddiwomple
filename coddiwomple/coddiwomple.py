"""
coddiwomple.py
A pythonic Sequential Monte Carlo (SMC) library for molecular mechanics and Bayesian inference

A compilation of functions (coddiwomple library wrappers) that are commonly used in practice
"""

#####Imports#####
import logging
import os
import sys
from pkg_resources import resource_filename
import random

#####Constants#####
quote_dict = {
                "Everything should be fine; you'll find things tend to stand in line": 'Beirut',
                "No": 'Rosa Parks',
                'Y la monotonia es un asesino lento': 'Monotonia, The Growlers',
                'Ich bin der zorn Gottes.  Die erde uber die ich gehe, sieht mich und bebt.': 'Aguirre, Der Zorn Gottes'
             }


def canvas(with_attribution=True):
    """
    let's play

    arguments
        with_attribution : bool, Optional, default: True
            Set whether or not to display who the quote is from

    return
        quote : str
            Compiled string including quote and optional attribution
    """
    quote = random.choice(list(quote_dict.keys()))
    if with_attribution:
        attribution = quote_dict[quote]
        quote += f"\n\t- {attribution}"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
