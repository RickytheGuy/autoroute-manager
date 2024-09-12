import logging
import sys

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.propagate = False
logging_format = logging.Formatter('%(asctime)s - %(levelname)s -%(message)s', "%Y-%m-%d %H:%M:%S")

_handler = logging.StreamHandler(sys.stdout)  # creates the handler
_handler.setLevel(logging.INFO)  # sets the handler info
_handler.setFormatter(logging_format)
LOG.addHandler(_handler)

__version__ = '0.1.0'