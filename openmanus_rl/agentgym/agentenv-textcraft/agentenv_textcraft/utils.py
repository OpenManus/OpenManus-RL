from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ItemTag:
    tag: str = None
    item_id: str = None

    @property
    def name(self):
        return self.item_id or self.tag


@dataclass
class ItemTagWithCount:
    item_tag: ItemTag
    count: int


@dataclass(frozen=True)
class Recipe:
    input_items: List[ItemTagWithCount]
    output_item: ItemTagWithCount

    @property
    def recipe_str(self):
        output_str = "craft {} {} using ".format(
            self.output_item.count, item_id_to_str(self.output_item.item_tag.name)
        )
        for input_itemtag_count in self.input_items:
            output_str += "{} {}, ".format(
                input_itemtag_count.count,
                item_id_to_str(input_itemtag_count.item_tag.name),
            )
        output_str = output_str[:-2]
        return output_str


class ActionFailed(Exception):
    pass


def item_id_to_str(item_id: str):
    return item_id.replace("minecraft:", "").replace("_", " ")
