"""
coddiwomple
A pythonic Sequential Monte Carlo (SMC) library for molecular mechanics and Bayesian inference
"""

# Add imports here
from .coddiwomple import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
