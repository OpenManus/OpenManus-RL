import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "webarena")
)

# webarena urls
os.environ["SHOPPING"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ["SHOPPING_ADMIN"] = (
    "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
)
os.environ["REDDIT"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ["GITLAB"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ["MAP"] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ["WIKIPEDIA"] = (
    "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
)
os.environ["HOMEPAGE"] = "PASS"

os.chdir("./webarena")

from .launch import launch
from .server import app
