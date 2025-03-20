from copy import deepcopy
import json
from math import ceil
import os
import random
from unittest import skip
from typing import List, Set, Dict

from numpy import rec

from .utils import ItemTag, ItemTagWithCount, Recipe, ActionFailed, item_id_to_str


class CraftingTree:

    def __init__(self, minecraft_dir):
        self.tag_recipes = {}  # recipes for tags (i.e. item types)
        self.itemid_recipes: Dict[str, list[Recipe]] = {}  # recipes for items
        self.tag_set = set()  # set of tags
        self.itemid_set = set()
        self.item_id_to_tag = {}  # mapping from item id to tag
        # all the items that could be used to craft an item (down to the base items). Useful to
        # remove cycles
        self.transitive_dependencies = {}
        # minimum depth of recipe tree to craft an item
        self.min_depth = {}
        self._load_recipes(minecraft_dir)
        self.clean_up_recipes()

    def clean_up_recipes(self):
        # make sure every recipe with input tag has craftable recipes or items
        new_items = set()
        for item, recipes in self.itemid_recipes.items():
            # for each recipe
            for recipe in recipes:
                # for each input item
                for input_item in recipe.input_items:
                    input_tag = input_item.item_tag.tag
                    # when only tag is specified
                    if input_item.item_tag.item_id is None:
                        assert input_tag is not None
                        # make sure that the tag is craftable or fetchable
                        item_list = list(self.get_items_with_tags(input_tag))
                        success = False
                        # if an item in this list has a recipe, we are good
                        for item_id in item_list:
                            if item_id in self.itemid_recipes:
                                success = True
                                break
                        # if not, this type can't be crafted, so we convert it to an item
                        if not success:
                            input_item.item_tag = ItemTag(item_id=input_tag)
                            new_items.add(input_tag)

        # clean up itemid_set and tag_set
        for item in new_items:
            self.itemid_set.add(item)
            self.tag_set.remove(item)

    def _load_recipes(self, minecraft_dir):
        for f in os.listdir(os.path.join(minecraft_dir, "recipes/")):
            with open(os.path.join(minecraft_dir, "recipes/", f), "r") as fp:
                recipe_details = json.load(fp)
                input_items = []
                if recipe_details["type"] == "minecraft:crafting_shaped":
                    pattern = recipe_details["pattern"]
                    for key, item in recipe_details["key"].items():
                        count = 0
                        if isinstance(item, list):
                            item = item[0]
                        for line in pattern:
                            count += line.count(key)
                        if "item" in item:
                            input_item = ItemTag(item_id=item["item"])
                            self.itemid_set.add(item["item"])
                        elif "tag" in item:
                            input_item = ItemTag(tag=item["tag"])
                            self.tag_set.add(item["tag"])
                        else:
                            print(recipe_details, item)
                            raise ValueError("Unknown item type")
                        input_items.append(ItemTagWithCount(input_item, count))
                elif recipe_details["type"] == "minecraft:crafting_shapeless":
                    item_name_idx = {}
                    for ingredient in recipe_details["ingredients"]:
                        if isinstance(ingredient, list):
                            ingredient = ingredient[0]
                        if "item" in ingredient:
                            item_name = ingredient["item"]
                            input_item = ItemTag(item_id=item_name)
                            self.itemid_set.add(item_name)
                        elif "tag" in ingredient:
                            input_item = ItemTag(tag=ingredient["tag"])
                            item_name = ingredient["tag"]
                            self.tag_set.add(ingredient["tag"])
                        else:
                            print(recipe_details)
                            raise ValueError("Unknown item type")
                        if item_name not in item_name_idx:
                            item_name_idx[item_name] = len(input_items)
                            input_items.append(ItemTagWithCount(input_item, 1))
                        else:
                            curr_count = input_items[item_name_idx[item_name]].count
                            # frozen dataclass so can't modify in place
                            input_items[item_name_idx[item_name]] = ItemTagWithCount(
                                input_item, curr_count + 1
                            )
                else:
                    continue

                recipe_result = recipe_details["result"]
                if isinstance(recipe_result, str):
                    output_item_id = recipe_result
                    output_item_count = 1
                elif "item" in recipe_result:
                    output_item_id = recipe_result["item"]
                    output_item_count = recipe_result.get("count") or 1
                else:
                    print(recipe_details)
                    raise ValueError("Unknown item type")
                self.itemid_set.add(output_item_id)
                # Remove block recipes
                if len(input_items) == 1 and input_items[0].item_tag.name.endswith(
                    "_block"
                ):
                    continue
                output_tag = None
                if "group" in recipe_details:
                    output_tag = "minecraft:" + recipe_details["group"]
                    # sometimes the group is the same as the output item id
                    if output_tag != output_item_id:
                        self.tag_set.add(output_tag)
                        self.item_id_to_tag[output_item_id] = output_tag

                output_item = ItemTagWithCount(
                    ItemTag(tag=output_tag, item_id=output_item_id), output_item_count
                )
                recipe = Recipe(input_items, output_item)

                if output_item_id not in self.transitive_dependencies:
                    self.transitive_dependencies[output_item_id] = set()
                skip_recipe = False
                for input_itemtag_count in input_items:
                    input_item_name = input_itemtag_count.item_tag.name
                    if input_item_name in self.transitive_dependencies:
                        if (
                            output_item_id
                            in self.transitive_dependencies[input_item_name]
                        ):
                            skip_recipe = True

                if not skip_recipe:
                    recipe_item_id = output_item.item_tag.item_id
                    if recipe_item_id not in self.itemid_recipes:
                        self.itemid_recipes[recipe_item_id] = [recipe]
                    else:
                        self.itemid_recipes[recipe_item_id].append(recipe)

                    for input_itemtag_count in input_items:
                        input_item_name = input_itemtag_count.item_tag.name
                        self.transitive_dependencies[output_item_id].add(
                            input_item_name
                        )
                        if input_item_name in self.transitive_dependencies:
                            self.transitive_dependencies[output_item_id].update(
                                self.transitive_dependencies[
                                    input_itemtag_count.item_tag.name
                                ]
                            )

                    recipe_tag = output_item.item_tag.tag
                    if recipe_tag is not None:
                        if recipe_tag not in self.tag_recipes:
                            self.tag_recipes[recipe_tag] = [recipe]
                        else:
                            self.tag_recipes[recipe_tag].append(recipe)

    def craft(self, recipe: Recipe) -> ItemTagWithCount:
        if recipe.output_item.item_tag.item_id not in self.itemid_recipes:
            return None
        target_recipes = self.itemid_recipes[recipe.output_item.item_tag.item_id]
        for target_recipe in target_recipes:
            success = True
            # check that input recipe items matches the target recipe items
            input_recipe_items_clone = deepcopy(recipe.input_items)
            for itemtag_count in target_recipe.input_items:
                itemtag = itemtag_count.item_tag
                input_itemtag_count = self.find_matching_item(
                    itemtag, input_recipe_items_clone
                )
                if input_itemtag_count is None:
                    success = False
                    break

                if input_itemtag_count.count != itemtag_count.count:
                    print("Wrong Item Count for: {}".format(input_itemtag_count))
                    success = False
                    break

                input_recipe_items_clone.remove(input_itemtag_count)

            if len(input_recipe_items_clone):
                success = False

            if success:
                return target_recipe.output_item

        return None

    def find_matching_item(
        self, itemtag: ItemTag, input_recipe_items: List[ItemTagWithCount]
    ):
        for input_itemtag_count in input_recipe_items:
            if itemtag.item_id is not None:
                if input_itemtag_count.item_tag.item_id == itemtag.item_id:
                    return input_itemtag_count
            elif itemtag.tag is not None:
                if (
                    input_itemtag_count.item_tag.tag == itemtag.tag
                    or self.item_id_to_tag.get(input_itemtag_count.item_tag.item_id)
                    == itemtag.tag
                ):
                    return input_itemtag_count
        return None

    def is_craftable(self, item: str):
        return item in self.itemid_recipes or item in self.tag_recipes

    def is_valid_item(self, item: str):
        return item in self.itemid_set

    def is_tag(self, input: str):
        return input in self.tag_set

    def get_items_with_tags(self, input_tag: str):
        for item_id, tag in self.item_id_to_tag.items():
            if input_tag == tag:
                yield item_id

    def print_all_recipes(self):
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                self.print_recipe(recipe)
        for tag, recipes in self.tag_recipes.items():
            for recipe in recipes:
                self.print_recipe(recipe)

    def print_recipe(self, recipe: Recipe):
        print(recipe.recipe_str)

    def traverse_recipe_tree(self, item_name: str, visited_names: Set[str]):
        if item_name in visited_names:
            print("Cycle detected for {}: {}".format(item_name, visited_names))
            return []
        recipes = (
            self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or []
        )
        for recipe in recipes:
            new_visited_names = deepcopy(visited_names)
            for input_itemtag_count in recipe.input_items:
                input_item_name = input_itemtag_count.item_tag.name
                new_visited_names.add(item_name)
                recipes.extend(
                    self.traverse_recipe_tree(input_item_name, new_visited_names)
                )
        return recipes

    def collect_item_uses(self):
        item_uses = {}
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    if input_itemtag.item_tag.name not in item_uses:
                        item_uses[input_itemtag.item_tag.name] = []

                    item_uses[input_itemtag.item_tag.name].append(recipe)
        for tag, recipes in self.tag_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    if input_itemtag.item_tag.name not in item_uses:
                        item_uses[input_itemtag.item_tag.name] = []
                    item_uses[input_itemtag.item_tag.name].append(recipe)
        return item_uses

    def get_min_depth(self, item_tag: str):
        if item_tag in self.min_depth:
            return self.min_depth[item_tag]

        if item_tag in self.itemid_recipes:
            self.min_depth[item_tag] = self.get_min_depth_recipes(
                self.itemid_recipes[item_tag]
            )
        elif item_tag in self.tag_recipes:
            self.min_depth[item_tag] = self.get_min_depth_recipes(
                self.tag_recipes[item_tag]
            )
        else:
            self.min_depth[item_tag] = 0

        return self.min_depth[item_tag]

    def get_min_depth_recipes(self, recipes):
        depths = []
        for recipe in recipes:
            recipe_depths = []
            for input_itemtag_count in recipe.input_items:
                recipe_depths.append(
                    self.get_min_depth(input_itemtag_count.item_tag.name) + 1
                )
            # pick the max here since each item has to be built
            depths.append(max(recipe_depths))
        # pick the min here since the model could make the easiest recip
        return min(depths)

    def item_recipes_min_depth(self, min_depth: int):
        for item, recipes in self.itemid_recipes.items():
            item_depth = self.get_min_depth(item)
            if item_depth >= min_depth:
                yield item, item_depth

    def item_recipes_min_items(self, min_items: int):
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                if len(recipe.input_items) >= min_items:
                    yield item

    def item_recipes_min_closure(self, min_closure: int):
        for item, closure in self.transitive_dependencies.items():
            if len(closure) >= min_closure:
                yield item

    def create_recipe_set(self, item_name: str):
        item_uses = self.collect_item_uses()
        recipes = self.traverse_recipe_tree(item_name, set())
        distractors = []
        for recipe in recipes:
            for item in recipe.input_items:
                input_item_name = item.item_tag.name
                if input_item_name in item_uses:
                    input_item_uses_recipes = item_uses[input_item_name]
                    distractors.extend(
                        random.sample(
                            input_item_uses_recipes,
                            min(len(input_item_uses_recipes), 10),
                        )
                    )

        return (recipes, distractors)


def main():
    tree = CraftingTree(minecraft_dir="agentenv_textcraft/")

    for item_id, recipes in tree.itemid_recipes.items():
        print(item_id, tree.get_min_depth(item_id), sep="\t")

    for item_tag, recipes in tree.tag_recipes.items():
        print(item_tag, tree.get_min_depth(item_tag), sep="\t")


if __name__ == "__main__":
    main()
