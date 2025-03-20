import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Toolusage")
)

from .academia_launch import launch
from .academia_server import app
