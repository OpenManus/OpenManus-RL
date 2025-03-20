import pytest
from web_agent_site.engine.normalize import *

def test_normalize_color():
    suite = [
        ("", ""),
        ("black forest", "black"),
        ("violet lavender", "lavender"),
        ("steelivy fuchsia", "fuchsia"),
        ("123alabaster", "alabaster"),
        ("webshop", "webshop")
    ]
    for color_string, expected in suite:
        output = normalize_color(color_string)
        assert type(output) is str
        assert output == expected

def test_normalize_color_size():
    product_prices = {
        (1, "black forest", "3 meter"): 10.29,
        (2, "violet lavender", "xx-large"): 23.42,
        (3, "steelivy fuchsia", "random value"): 193.87,
        (4, "123alabaster", "40cm plus"): 67.23,
        (5, "webshop", "142"): 1.02,
        (6, "webshopsteel", "2 petite"): 57.99,
        (7, "leather black", "91ft walnut feet"): 6.20,
    }
    color_mapping_expected = {
        'N.A.': 'not_matched',
        "black forest": "black",
        "violet lavender": "lavender",
        "steelivy fuchsia": "fuchsia",
        "123alabaster": "alabaster",
        "webshop": "not_matched",
        "webshopsteel": "steel",
        "leather black": "black"
    }
    size_mapping_expected = {
        'N.A.': 'not_matched',
        "3 meter": '(.*)meter',
        "xx-large": 'xx-large',
        "random value": "not_matched",
        "40cm plus": '(.*)plus',
        "142": "numeric_size",
        "2 petite": "(.*)petite",
        "91ft walnut feet": '(.*)ft',
    }

    color_mapping, size_mapping = normalize_color_size(product_prices)
    assert type(color_mapping) == dict
    assert type(size_mapping)  == dict
    assert color_mapping == color_mapping_expected
    assert size_mapping  == size_mapping_expected
