""" flimview: small set of utilities to access and analyse FLIM data"""
__author__ = "Matias Carrasco Kind"
__license__ = "NCSA"
from .version import __version__

from . import flim, io_utils, models, plot_utils, datasets
__all__ = ["flim", "io_utils", "models", "plot_utils", "datasets"]
