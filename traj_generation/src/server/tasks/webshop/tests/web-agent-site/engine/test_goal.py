import pytest
from math import isclose
from web_agent_site.engine.goal import *

def test_get_type_reward():
    # Exact Match
    goal = {
        'query': "Query 1",
        'product_category': "a › b › c",
        'name': "Name 1"
    }
    purchased = {
        'query': "Query 1",
        'product_category': "a › b › c",
        'name': "Name 1"
    }
    result = get_type_reward(purchased, goal)
    assert result['r_type'] == 1.
    assert result['query_match'] == True
    assert result['category_match'] == True
    assert result['title_score'] == 1

    # Query Mismatch
    purchased['query'] = 'Query 2'
    result = get_type_reward(purchased, goal)
    assert result['query_match'] == False

    # Out of order / non-matching / partially matching / duplicate categories
    purchased['product_category'] = "b › c › a"
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = "d › e › f"
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == False
    purchased['product_category'] = "a › d › b"
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = "a › a › b"
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == True
    purchased['product_category'] = "a › a › d"
    result = get_type_reward(purchased, goal)
    assert result['category_match'] == False

    # Similar product names
    goal['name'] = "Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White"
    purchased['name'] = "PEAK High Top Mens Basketball Shoes Lou Williams Streetball Master Breathable Non Slip Outdoor Sneakers"
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.333, abs_tol=1e-2)

    # Slightly similar product names
    goal['name'] = "Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar Home Theater System Subwoofer"
    purchased['name'] = "BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System"
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.3, abs_tol=1e-2)

    goal['name'] = "Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar"
    purchased['name'] = "BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System"
    result = get_type_reward(purchased, goal)
    assert isclose(result['title_score'], 0.15, abs_tol=1e-2)

    # Completely different product names
    goal['name'] = "Rusticware 921ORB Kitchen and Bath Cabinet Knob"
    purchased['name'] = "Minkissy 2pcs Stainless Steel Eyebrow Tweezers Blackhead Acne Remover Portable Makeup Tweezers (Silver)"
    result = get_type_reward(purchased, goal)
    assert result['title_score'] < 0.05

def test_get_attribute_reward():
    # Exact Match
    goal = {
        'attributes': ["tea tree", "essential oils", "natural ingredients"],
    }
    purchased = {
        'Attributes': ["tea tree", "essential oil", "natural ingredients"]
    }
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 1
    assert num_attr_matches == 3

    # Partial Match
    goal = {
        'attributes': ["tea tree", "essential oils", "natural ingredients"]
    }
    purchased = {
        'Attributes': ["essential oil", "natural ingredients"],
        'Title': "",
        'BulletPoints': [],
        'Description': ""
    }
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 2./3.
    assert num_attr_matches == 2

    # Goal attributes found in purchased non-goals
    goal = {
        'attributes': ["tea tree", "essential oils", "natural ingredients"]
    }
    purchased = {
        'Attributes': [],
        'Title': "",
        'BulletPoints': ["This shampoo has essential oils and smells like lemons"],
        'Description': "Best shampoo on the market, made with natural ingredients"
    }
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 2./3.
    assert num_attr_matches == 2

    # No match
    goal = {
        'attributes': ["tea tree", "essential oils", "natural ingredients"]
    }
    purchased = {
        'Attributes': ["tea bag", "earl gray", "lipton"],
        'Title': "English tea for breakfast",
        'BulletPoints': ["Soothing aroma", "Calming, great feeling"],
        'Description': "Best tea made by Lipton, great to pair with breakfast"
    }
    r_attr, num_attr_matches = get_attribute_reward(purchased, goal)
    assert r_attr == 0
    assert num_attr_matches == 0

def test_get_option_reward():
    # Exact Match
    goal = ["grey", "XL", "pack of 12"]
    purchased = ["pack of 12", "grey", "XL"]
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal)
    assert r_option == 1

    # Partial Match
    goal = ["grey", "XL", "pack of 12"]
    purchased = ["pack of 12", "blue", "XL"]
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal) - 1
    assert r_option == 2./3.

    # Fuzzy Match
    goal = ["cool powder snow", "XL", "pack of 12"]
    purchased = ["pack of 12", "powder snow", "XL"]
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal)
    assert r_option == 1

    # Empty Goal Options
    goal = []
    purchased = ["goal 1", "goal 2"]
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == 0
    assert r_option == None

    # Empty Purchased Options
    goal = ["goal 1", "goal 2"]
    purchased = []
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == 0
    assert r_option == 0

def test_get_reward():
    # Exact Match
    goal = {
        'query': "Query 1",
        'product_category': "a › b › c",
        'name': "Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White",
        'attributes': ["tea tree", "essential oils", "natural ingredients"],
        'goal_options': {"color": "grey", "size": "XL"},
        'price_upper': 40.00
    }
    purchased = {
        'query': "Query 1",
        'product_category': "a › b › c",
        'name': "Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White",
        'Attributes': ["tea tree", "essential oil", "natural ingredients"],
        'Title': "",
        'BulletPoints': [],
        'Description': "",
        'goal_options': {"color": "grey", "size": "XL"}
    }
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert total_reward == 1

    # Variation in r_attributes reward
    purchased['Attributes'] = []
    purchased['Title'] = ""
    purchased['BulletPoints'] = "This shampoo has essential oils and smells like lemons"
    purchased['Description'] = "Best shampoo on the market, made with natural ingredients"
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 2./3., abs_tol=1e-2)

    # Variation in r_option reward
    goal['goal_options'] = {"color": "grey", "size": "XL", "amount": "pack of 12"}
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 0.5714, abs_tol=1e-2)

    # Variation in r_type reward
    goal['name'] = "Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar"
    purchased['name'] = "BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System"
    purchased['query'] = "Query 2"
    purchased['product_category'] = "a › d › e"
    total_reward = get_reward(purchased, goal, 35, purchased['goal_options'])
    assert isclose(total_reward, 0.2857, abs_tol=1e-2)