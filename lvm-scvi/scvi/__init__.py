# -*- coding: utf-8 -*-

"""Top-level package for scVI-dev."""

__author__ = "Romain Lopez"
__email__ = "romain_lopez@berkeley.edu"
__version__ = '0.3.0'

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
logging.basicConfig(level=logging.INFO)
