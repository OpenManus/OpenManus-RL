import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "webshop")
)

from .launch import launch
from .server import app
