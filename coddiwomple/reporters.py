"""
Reporter Module
"""
#####Imports#####
import copy
import logging
import numpy as np

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("reporters")
_logger.setLevel(logging.DEBUG)

class Reporter():
    """
    Generalized reporter object for Particles
    """
    def __init__(self, **kwargs):
        """
        Generalized Reporter Init Method.
        """
        pass
