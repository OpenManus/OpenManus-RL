import json

fish_description = {
    "spray": {
        "passive": r"Counter: Deal 30 damage to attacker when a teammate's health is below 30%. ",
        "active": r"AOE: Attack all enemies for 35% of its attack points."
    },
    "flame": {
        "passive": r"Counter: Deal 30 damage to attacker when a teammate's health is below 30%. ",
        "active": r"Infight: Attack one alive teammate for 75 damage and increases your attack points by 140. Notice! You can't attack yourself or dead teamate! " # 
    },
    "eel": {
        "passive": r"Deflect: Distribute 70% damage to teammates and takes 30% when attacked. Gains 40 attack points after taking 200 damage accumulated. ",
        "active": r"AOE: Attack all enemies for 35% of your attack points."
    },
    "sunfish": {
        "passive": r"Deflect: Distribute 70% damage to teammates and takes 30% when attacked. Gains 40 attack points after taking 200 damage accumulated. ",
        "active": r"Infight: Attack one alive teammate for 75 damage and increases your attack points by 140. Notice! You can't attack yourself or dead teamate! " #
    },
    "barracuda": {
        "passive": r"Reduce: There is a 30% chance to avoid any incoming damage each time. ",
        "active": r"Crit: Deals 120 CRITICAL damage to an enemy. " # FIXME:
    },
    "mobula": {
        "passive": r"Reduce: There is a 30% chance to avoid any incoming damage each time. ",
        "active": r"Subtle: Choose a teammate or yourself to reduce the damage taken by 70% when attacked, and increase its attack points by 20." # = Subtle: dont explicitly display
    },
    "octopus": {
        "passive": r"Heal: Regain 20 health points if the health is still greater than 0 when attacked. ",
        "active": r"Subtle: Choose a teammate or yourself to reduce the damage taken by 70% when attacked, and increase its attack points by 20."
    },
    "whiteshark": {
        "passive": r"Heal: Regain 20 health points if the health is still greater than 0 when attacked. ",
        "active": r"Crit: Deal 120% CRITICAL damage of your attack power to the enemy with the lowest health. If the target's health is below 160, increase the CRITICAL damage to 140%. "
    },
    "hammerhead": {
        "passive": r"Explode: Deal 40 damage to the source when attacked but not died. When the health is below 20%, increase its attack points by 15. ",
        "active": r"Crit: Deal 120% CRITICAL damage of your attack power to the enemy with the lowest health. If the target's health is below 160, increase the CRITICAL damage to 140%. "
    },
}

target_friend = ["flame", "eel", "mobula", "octopus"]

action_format = {
    "pick_fish": "pick an alive fish, you should give the name of the alive fish",
    "action": "choose from [normal, active]",
    "target_position": "target's position, you must choose from [0,3]"
}

guess_format = {
    "guess_type": "the enemy's fish type you may guess",
    "target_position": "guess target's position, you must choose from [0,3]"
}

ACTION_PROMPT = """This is a two-player battle game with four pet fish on each team. The types of fish may vary.
Each fish has its 400 initial health, 200 attack power, active ability, and passive ability.
You can choose a live fish to use its active skill or normal attack (causing half of attack power as damage) on an enemy fish each round.
When the conditions are met, the fish's passive ability will automatically trigger, regardless of whether it is chosen.
%s
The victory condition is to have more fish alive at the end of the game.

The following are the four types of your pet fish:
%s

The following are the four types of enemy's pet fish:
%s

Play the game with me. In each round, you should output your thinking process, and return your move with following JSON format:
%s

Notice! You must return your move in each round. Otherwise, you will be considered defeated."""

GUESS_RULE = """Your fish's identity is initially hidden. The enemy can guess one of your fish's identity in each round. If the enemy guesses right, your fish's identity is revealed, and each of your fish will get 50 damage."""

GUESS_PROMPT = """This is a two-player battle game with four pet fish in each team. The types of fish may vary.
Each fish has its initial health, attack power, active ability, and passive ability.
All fish\'s identities are initially hidden. You should guess one of the enemy fish\'s identities in each round. If you guess right, the enemy fish\'s identity is revealed, and each of the enemy's fish will get 50 damage. You can only guess the identity of the live fish.
The victory condition is to have more fish alive at the end of the game.

The following are the four types of your pet fish:
%s

The following are the four types of enemy's pet fish:
%s

Play the game with me. In each round, you should output your thinking process, and return your move with following JSON format:
%s

Notice! You must return your move in each round. Otherwise, you will be considered defeated."""

your_fish = ["spray", "flame", "eel", "sunfish"] # ["mobula", "octopus", "whiteshark", "hammerhead"]

enemy_fish = ["spray", "flame", "eel", "sunfish"] # ["mobula", "octopus", "whiteshark", "hammerhead"]

def get_fish_description(fish):
    desc = {}
    for i in fish:
        desc[i] = fish_description[i]
    return desc

your_fish_description = get_fish_description(your_fish)
enemy_fish_description = get_fish_description(enemy_fish)

ACTION_DESCRIPTION = {
    1: ACTION_PROMPT % ('', your_fish_description, enemy_fish_description, action_format),
    2: ACTION_PROMPT % (GUESS_RULE, your_fish_description, enemy_fish_description, action_format)
}

GUESS_DESCRIPTION = {
    2: GUESS_PROMPT % (your_fish_description, enemy_fish_description, guess_format),
}

action_prompt = """
Following is the current game state:
%s
Your previous action: 
%s
Enemy's previous action:
%s
Please Output your next action.
"""

guess_prompt = """
Previous Guess: 
%s
Live Unknown Enemy Fish: 
%s
Enemy's previous action: 
%s
Enemy's previous triggered passive ability: 
%s
Please output your guess.
"""         