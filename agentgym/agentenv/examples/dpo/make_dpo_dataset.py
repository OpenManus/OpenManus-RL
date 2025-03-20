import argparse
import json

from tqdm import tqdm


def load_data(file_path):
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf8") as f:
            data = [json.loads(line) for line in f]
    else:
        assert file_path.endswith(".json")
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
    print(f"Loaded {len(data)} data from {file_path}")
    return data


def make_dpo_data(chosen, rejected, prompt_length):
    return {
        "id": int(chosen["item_id"].split("_")[-1]),
        "prompt": chosen["conversations"][:prompt_length],
        "chosen": chosen["conversations"][prompt_length:],
        "rejected": rejected["conversations"][prompt_length:],
    }


def main(args):
    expert = load_data(args.expert)
    experience = load_data(args.experience)

    if args.task is not None:
        expert = [x for x in expert if x["item_id"].startswith(args.task)]
        experience = [x for x in experience if x["item_id"].startswith(args.task)]

    # deduplicate
    new_expert = {}
    for x in expert:
        idx = x["item_id"]
        if idx not in new_expert:
            new_expert[idx] = x
        elif x["reward"] < new_expert[idx]["reward"]:
            new_expert[idx] = x
    expert = list(new_expert.values())

    new_experience = {}
    for x in experience:
        idx = x["item_id"]
        if idx not in new_experience:
            new_experience[idx] = x
        elif x["reward"] < new_experience[idx]["reward"]:
            new_experience[idx] = x
    experience = list(new_experience.values())

    if len(expert) != len(experience):
        raise ValueError(
            f"Length of expert ({len(expert)}) and experience ({len(experience)}) are different."
        )

    expert.sort(key=lambda x: x["item_id"])
    experience.sort(key=lambda x: x["item_id"])

    print(f"Length of expert and experience: {len(expert)}")

    dpo_data = []
    for e, x in tqdm(zip(expert, experience)):
        if e["item_id"] != x["item_id"]:
            raise ValueError(f"Item ID of expert ({e['item_id']}) and experience ({x['item_id']}) are different.")
        reward_expert = e.get("reward", 1.0)
        reward_experience = x["reward"]
        if (
            reward_expert - reward_experience >= args.reward_gap
            and reward_expert >= args.expert_threshold
        ):
            dpo_data.append(make_dpo_data(e, x, args.prompt_length))
        elif (
            reward_experience - reward_expert >= args.reward_gap
            and reward_experience >= args.expert_threshold
        ):
            dpo_data.append(make_dpo_data(x, e, args.prompt_length))

    print(f"Length of DPO dataset: {len(dpo_data)}")

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(dpo_data, f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=str, required=True)
    parser.add_argument("--prompt_length", type=int, default=3)
    parser.add_argument("--experience", type=str, required=True)
    parser.add_argument("--reward_gap", type=float, default=0.01)
    parser.add_argument("--expert_threshold", type=float, default=0.7)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--task", type=str, required=False)
    main(parser.parse_args())
