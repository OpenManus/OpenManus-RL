import json
import re
from collections import defaultdict
from typing import Any, TypedDict, Union

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from playwright.sync_api import CDPSession, Page, ViewportSize

from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    UTTERANCE_MAX_LENGTH,
)

from .utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
    DOMTree,
    Observation,
    png_bytes_to_numpy,
)

IN_VIEWPORT_RATIO_THRESHOLD = 0.6


class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }


class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds

        # extract browser info
        win_top_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_top_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_top_bound": win_top_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @staticmethod
    def get_bounding_client_rect(
        client: CDPSession, backend_node_id: str
    ) -> dict[str, Any]:
        try:
            remote_object = client.send(
                "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
            )
            remote_object_id = remote_object["object"]["objectId"]
            response = client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as e:
            return {"result": {"subtype": "error"}}

    @staticmethod
    def get_element_in_viewport_ratio(
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: BrowserConfig,
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = config["win_width"]
        win_top_bound = 0
        win_lower_bound = config["win_height"]

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound)
            - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound)
            - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / width * height
        return ratio

    def fetch_page_html(
        self,
        info: BrowserInfo,
        page: Page,
        client: CDPSession,
        current_viewport_only: bool,
    ) -> DOMTree:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]

        # make a dom tree that is easier to navigate
        dom_tree: DOMTree = []
        graph = defaultdict(list)
        for node_idx in range(len(nodes["nodeName"])):
            cur_node: DOMNode = {
                "nodeId": "",
                "nodeType": "",
                "nodeName": "",
                "nodeValue": "",
                "attributes": "",
                "backendNodeId": "",
                "parentId": "",
                "childIds": [],
                "cursor": 0,
                "union_bound": None,
            }

            node_type_idx = nodes["nodeType"][node_idx]
            node_type = "generic"
            if node_type_idx >= 0 and node_type_idx < len(strings):
                node_type = strings[node_type_idx]

            node_name = strings[nodes["nodeName"][node_idx]]

            node_value_idx = nodes["nodeValue"][node_idx]
            node_value = ""
            if node_value_idx >= 0 and node_value_idx < len(strings):
                node_value = " ".join(strings[node_value_idx].split())

            node_attributes = [
                strings[i] for i in nodes["attributes"][node_idx]
            ]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            cur_node["nodeId"] = str(node_idx)
            cur_node["nodeType"] = node_type
            cur_node["nodeName"] = node_name
            cur_node["nodeValue"] = node_value
            cur_node["attributes"] = node_attributes_str
            cur_node["backendNodeId"] = str(nodes["backendNodeId"][node_idx])
            cur_node["parentId"] = str(nodes["parentIndex"][node_idx])

            if cur_node["parentId"] != "-1":
                graph[cur_node["parentId"]].append(str(cur_node["nodeId"]))

            # get the bound
            if cur_node["parentId"] == "-1":
                cur_node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client, cur_node["backendNodeId"]
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    cur_node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    cur_node["union_bound"] = [x, y, width, height]

            dom_tree.append(cur_node)

        # add parent children index to the node
        for parent_id, child_ids in graph.items():
            dom_tree[int(parent_id)]["childIds"] = child_ids

        # remove the nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: DOMNode) -> None:
                # update the node information in the accessibility tree
                node_id = node["nodeId"]
                parent_id = node["parentId"]
                child_ids = node["childIds"]

                # update the children of the parent node
                assert dom_tree[int(parent_id)]["parentId"] != "[REMOVED]"
                # remove the nodeid from parent
                index = dom_tree[int(parent_id)]["childIds"].index(node_id)
                dom_tree[int(parent_id)]["childIds"].pop(index)

                # Insert children_nodeids in the same location
                for child_id in child_ids:
                    dom_tree[int(parent_id)]["childIds"].insert(
                        index, child_id
                    )
                    index += 1

                # update children node's parent
                for child_id in child_ids:
                    dom_tree[int(child_id)]["parentId"] = parent_id
                # mark as removed
                dom_tree[int(node_id)]["parentId"] = "[REMOVED]"

            config = info["config"]
            for cursor, node in enumerate(dom_tree):
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0.0 or height == 0.0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            dom_tree = [
                node
                for node in dom_tree
                if node.get("parentId", "-1") != "[REMOVED]"
            ]

        return dom_tree

    @staticmethod
    def parse_html(dom_tree: DOMTree) -> tuple[str, dict[str, Any]]:
        """Parse the html tree into a string text"""

        obs_nodes_info = {}
        nodeid_to_cursor = {
            node["nodeId"]: idx for idx, node in enumerate(dom_tree)
        }

        def dfs(node_cursor: int, depth: int) -> str:
            tree_str = ""
            node = dom_tree[node_cursor]
            indent = "\t" * depth
            valid_node = True
            try:
                node_str = f"[{node_cursor}] <{node['nodeName']}"
                if node["attributes"]:
                    node_str += f" {node['attributes']}"
                node_str += f"> {node['nodeValue']}"
                valid_node = bool(node["attributes"] or node["nodeValue"])

                if valid_node:
                    obs_nodes_info[str(node_cursor)] = {
                        "backend_id": node["backendNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }
                    tree_str += f"{indent}{node_str}\n"

            except Exception as e:
                valid_node = False

            for child_ids in node["childIds"]:
                child_cursor = nodeid_to_cursor[child_ids]
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(child_cursor, child_depth)
                tree_str += child_str

            return tree_str

        html = dfs(0, 0)
        return html, obs_nodes_info

    def fetch_page_accessibility_tree(
        self,
        info: BrowserInfo,
        client: CDPSession,
        current_viewport_only: bool,
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            backend_node_id = str(node["backendDOMNodeId"])
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client, backend_node_id
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]

        # filter nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                # update the children of the parent node
                assert (
                    accessibility_tree[parent_cursor].get("parentId", "Root")
                    is not None
                )
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(
                    nodeid
                )
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(
                        index, child_nodeid
                    )
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor][
                        "parentId"
                    ] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            config = info["config"]
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            accessibility_tree = [
                node
                for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]

        return accessibility_tree

    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            # remove statictext if the content already appears in the previous line
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText (.+)"

                match = re.search(pattern, line, re.DOTALL)
                if match:
                    static_text = match.group(1)[1:-1]  # remove the quotes
                    if static_text and all(
                        static_text not in prev_line
                        for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    def process(self, page: Page, client: CDPSession) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=5000)
            browser_info = self.fetch_browser_info(page, client)

        if self.observation_type == "html":
            dom_tree = self.fetch_page_html(
                browser_info,
                page,
                client,
                current_viewport_only=self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_html(dom_tree)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                browser_info,
                client,
                current_viewport_only=self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_accessibility_tree(
                accessibility_tree
            )
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        else:
            raise ValueError(
                f"Invalid observatrion type: {self.observation_type}"
            )

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["union_bound"]
        x, y, width, height = node_bound
        center_x = x + width / 2
        center_y = y + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(self, observation_type: str):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.meta_data = create_empty_metadata()

    def process(self, page: Page, client: CDPSession) -> npt.NDArray[np.uint8]:
        try:
            screenshot = png_bytes_to_numpy(page.screenshot())
        except:
            page.wait_for_event("load")
            screenshot = png_bytes_to_numpy(page.screenshot())
        return screenshot


class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type, current_viewport_only, viewport_size
        )
        self.image_processor = ImageObservationProcessor(
            image_observation_type
        )
        self.viewport_size = viewport_size

    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    def get_observation(
        self, page: Page, client: CDPSession
    ) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page, client)
        image_obs = self.image_processor.process(page, client)
        return {"text": text_obs, "image": image_obs}

    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")
