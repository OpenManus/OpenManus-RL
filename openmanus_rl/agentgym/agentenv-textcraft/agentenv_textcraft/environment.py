import random
import gymnasium as gym
import re
from .utils import ActionFailed, ItemTag, ItemTagWithCount, Recipe, item_id_to_str
from .crafting_tree import CraftingTree
from typing import List


class TextCraftEnv(gym.Env[str, str]):
    def __init__(self, crafting_tree, commands, goal):
        self.inventory = {}
        self.action_regexes = {
            "craft": r"craft (.*) using (.*)",
            "get": r"get ([0-9]+) (.*)",
            "inventory": r"inventory",
        }
        self.count_regex = r"([0-9]+) (.*)"
        self.crafting_tree = crafting_tree
        self.commands = commands
        self.goal = goal

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        try:
            for action_type, regex in self.action_regexes.items():
                match = re.match(regex, action)
                if match:
                    if action_type == "craft":
                        recipe = self.extract_recipe(match.group(1), match.group(2))
                        if recipe is None:
                            raise ActionFailed(
                                "Could not extract recipe from {}".format(action)
                            )
                        if not self.has_items(recipe.input_items):
                            raise ActionFailed(
                                "Could not find enough items to craft {}".format(
                                    recipe.output_item.item_tag.item_id
                                )
                            )
                        output_itemtag_count = self.crafting_tree.craft(recipe)
                        if output_itemtag_count is None:
                            raise ActionFailed(
                                "Could not find a valid recipe for {}".format(
                                    recipe.output_item
                                )
                            )
                        self.remove_items(recipe.input_items)
                        self.add_item(
                            output_itemtag_count.item_tag, output_itemtag_count.count
                        )
                        observation = "Crafted {} {}".format(
                            output_itemtag_count.count,
                            output_itemtag_count.item_tag.item_id,
                        )
                        if output_itemtag_count.item_tag.item_id == self.goal:
                            reward = 1
                            terminated = True
                    elif action_type == "get":
                        (item, amt) = match.group(2), int(match.group(1))
                        item_obj = self.item_str_to_obj(item)
                        if self.crafting_tree.is_craftable(item_obj.name):
                            raise ActionFailed("Could not find {}".format(item))
                        if (
                            self.crafting_tree.is_tag(item_obj.item_id)
                            or item_obj.item_id is None
                        ):
                            raise ActionFailed("Could not find {}".format(item))
                        if not self.crafting_tree.is_valid_item(item_obj.item_id):
                            raise ActionFailed("Could not find {}".format(item))
                        self.add_item(item_obj, amt)
                        observation = "Got {} {}".format(amt, item)
                        if item_obj.item_id == self.goal:
                            reward = 1
                            terminated = True
                    elif action_type == "inventory":
                        observation = "Inventory: "
                        if not len(self.inventory.items()):
                            observation += "You are not carrying anything."
                        for item, amt in self.inventory.items():
                            observation += "[{}] ({}) ".format(
                                item_id_to_str(item), amt
                            )
                    else:
                        raise NotImplementedError(
                            "Action type {} not implemented".format(action_type)
                        )
            if observation is None:
                raise ActionFailed("Could not execute {}".format(action))

        except ActionFailed as e:
            observation = "{}".format(e.args[0])
            reward = 0
            info = {}

        return (observation, reward, terminated, truncated, info)

    def has_items(self, items: List[ItemTagWithCount]):
        for itemtag_count in items:
            if (
                itemtag_count.item_tag.item_id not in self.inventory
                or self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count
            ):
                return False
        return True

    def add_item(self, item_tag: ItemTag, amt: int):
        if item_tag.item_id not in self.inventory:
            self.inventory[item_tag.item_id] = 0
        self.inventory[item_tag.item_id] += amt

    def remove_items(self, items: List[ItemTagWithCount]):
        for itemtag_amts in items:
            self.inventory[itemtag_amts.item_tag.item_id] -= itemtag_amts.count
            if self.inventory[itemtag_amts.item_tag.item_id] == 0:
                del self.inventory[itemtag_amts.item_tag.item_id]

    def extract_recipe(self, output_item_str, input_items_str) -> Recipe:
        # check if there is a number in the output item
        m = re.match("([0-9]+) (.*)", output_item_str)
        if m:
            output_item = self.item_str_to_obj(m.group(2))
            output_item_count = int(m.group(1))
        else:
            output_item = self.item_str_to_obj(output_item_str)
            output_item_count = 1
        output_item_count = ItemTagWithCount(output_item, output_item_count)
        input_items = []
        for input_item_count in input_items_str.split(","):
            match = re.match(self.count_regex, input_item_count.strip())
            if match:
                count = int(match.group(1))
                item_str = match.group(2)
                input_item_obj = self.item_str_to_obj(item_str)
                input_items.append(ItemTagWithCount(input_item_obj, count))
            else:
                raise ActionFailed(
                    "Wrong item format: {}".format(input_item_count.strip())
                )
        return Recipe(input_items=input_items, output_item=output_item_count)

    def item_str_to_obj(self, item):
        item_id = "minecraft:" + item.replace(" ", "_")
        if self.crafting_tree.is_tag(item_id):
            return ItemTag(tag=item_id)
        else:
            return ItemTag(item_id=item_id)

    def reset(self, seed=42, data_idx=0, commands=None, goal=None):
        super().reset(seed=seed)
        # clean inventory
        self.inventory = {}
        if commands is not None and goal is not None:
            self.commands = commands
            self.goal = goal
            return (
                "Crafting commands:\n{}\n\nGoal: craft {}.".format(
                    self.commands, item_id_to_str(self.goal)
                ),
                {},
            )
        random.seed(seed)
        item_depth_list = list(self.crafting_tree.item_recipes_min_depth(1))
        # use idx to deterministically select goal
        sorted_item_depth_list = sorted(item_depth_list, key=lambda x: x[1])
        goal_depth = sorted_item_depth_list[data_idx % len(item_depth_list)]
        # example: self.goal = "minecraft:dark_oak_sign"
        self.goal = goal_depth[0]
        recipes_set = set()
        distractor_set = set()
        max_distractor = 10
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal)
        for recipe in recipes:
            recipes_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipes_set:
                distractor_set.add(distractor.recipe_str)

        recipes_list = list(recipes_set) + random.sample(
            list(distractor_set), min(len(distractor_set), max_distractor)
        )
        random.shuffle(recipes_list)
        self.commands = "\n".join(recipes_list)
        return (
            "Crafting commands:\n{}\n\nGoal: craft {}.".format(
                self.commands, item_id_to_str(self.goal)
            ),
            {},
        )

    def render(self, mode="human"):
        pass

    def close(self):
        pass
