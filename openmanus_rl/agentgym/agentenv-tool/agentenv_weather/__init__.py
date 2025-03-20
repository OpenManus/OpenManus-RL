import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Toolusage")
)

from .weather_launch import launch
from .weather_server import app
