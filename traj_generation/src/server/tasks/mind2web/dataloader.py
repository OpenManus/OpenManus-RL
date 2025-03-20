import pathlib
import random
import sys

import lxml
from datasets import load_dataset
from torch.utils.data import Dataset

sys.path.append(pathlib.Path(__file__).parent.parent.absolute().as_posix())
from .data_utils.dom_utils import get_tree_repr, prune_tree


def format_input_generation(
    sample, candidate_ids, gt=-1, previous_k=5, keep_html_brackets=False
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    if len(sample["previous_actions"]) > 0:
        for action in sample["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action?"
        "Please select the element to interact with, and the action to perform along with the value to type in or select. "
        "If the task cannot be completed, output None."
    )

    if gt == -1:
        seq_target = "None"
    else:
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"Element: {choices[gt][1]}\n"
        seq_target += f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices


def format_input_multichoice(
    sample, candidate_ids, gt=-1, previous_k=5, keep_html_brackets=False
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    if len(sample["previous_actions"]) > 0:
        for action in sample["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"
    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices


class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        neg_ratio=5,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
        top_k=-1,
    ):
        self.data = data
        self.neg_ratio = neg_ratio
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.mode = mode
        self.top_k = top_k

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, idx):
        sample = self.data[idx // 10]
        if self.top_k > 0:
            top_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] < self.top_k
            ]
            other_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] >= self.top_k
            ]
        else:
            top_negatives = []
            other_negatives = sample["neg_candidates"]
        if random.random() < 0.8 and len(top_negatives) > 0:
            neg_candidates = top_negatives
        else:
            neg_candidates = other_negatives

        if len(sample["pos_candidates"]) != 0 and (
            random.random() > self.neg_ratio or len(neg_candidates) == 0
        ):
            pos_candidate = random.choice(sample["pos_candidates"])
            neg_candidate = random.sample(
                neg_candidates,
                min(len(neg_candidates), self.num_candidates - 1),
            )
            gt = pos_candidate["backend_node_id"]
            candidate_ids = [gt] + [c["backend_node_id"] for c in neg_candidate]
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample, candidate_ids, gt
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample, candidate_ids, gt
                )
        else:
            neg_candidate = random.sample(
                neg_candidates,
                min(len(neg_candidates), self.num_candidates),
            )
            gt = -1
            candidate_ids = [c["backend_node_id"] for c in neg_candidate]
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample, candidate_ids, gt
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample, candidate_ids, gt
                )
        seq_context = self.tokenizer(
            seq_context,
            truncation=True,
            max_length=self.max_context_len,
            add_special_tokens=False,
        )
        seq_in = self.tokenizer(
            seq_in,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_context_len,
        )
        model_input = {
            "input_ids": seq_context["input_ids"] + seq_in["input_ids"],
            "attention_mask": seq_context["attention_mask"] + seq_in["attention_mask"],
        }
        seq_out = self.tokenizer(seq_out)
        model_input["labels"] = seq_out["input_ids"]
        return model_input


def get_data_split(
    data_dir,
    split_file,
    candidate_results=None,
    cache_dir=None,
    is_train=False,
    is_debug=False,
):
    def flatten_actions(samples):
        outputs = {
            "website": [],
            "confirmed_task": [],
            "annotation_id": [],
            "previous_actions": [],
            "action_uid": [],
            "operation": [],
            "pos_candidates": [],
            "neg_candidates": [],
            "cleaned_html": [],
        }
        num_actions = [len(actions) for actions in samples["actions"]]
        for key in ["website", "confirmed_task", "annotation_id"]:
            for idx, value in enumerate(samples[key]):
                outputs[key] += [value] * num_actions[idx]
        for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
            for a_idx, action in enumerate(actions):
                outputs["previous_actions"].append(action_reprs[:a_idx])
                for key in [
                    "action_uid",
                    "operation",
                    "pos_candidates",
                    "neg_candidates",
                    "cleaned_html",
                ]:
                    outputs[key].append(action[key])
        return outputs

    dataset = load_dataset(
        data_dir, data_files=split_file, split="all", cache_dir=cache_dir
    )
    if is_debug:
        dataset = dataset.select(range(3))
    flatten_dataset = dataset.map(
        flatten_actions,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=10,
        num_proc=4,
    )
    if candidate_results is not None:
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]

        def get_score(sample):
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            for candidates in [sample["pos_candidates"], sample["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]
            return {
                "pos_candidates": sample["pos_candidates"],
                "neg_candidates": sample["neg_candidates"],
            }

        flatten_dataset = flatten_dataset.map(get_score)
    if is_train:
        flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)

    return flatten_dataset
