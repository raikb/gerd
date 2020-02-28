import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gerd import models  # noqa: F401
from gerd import input_check  # noqa: F401
