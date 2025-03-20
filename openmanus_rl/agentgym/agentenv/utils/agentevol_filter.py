import argparse
import json
import os

import jsonlines

task_list = [
    "webshop",
    "alfworld",
    "textcraft",
    "sciworld",
    "sqlgym",
    "lmrlgym_wordle",
    "lmrlgym_maze",
    "weather",
    "movie",
    "todo",
    "babyai",
]
threshold_list = [0.99, 0.99, 0.99, 99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0]


def extract_category(item_id):
    for i, task in enumerate(task_list):
        if item_id.startswith(task):
            return task
    return None


def filter_jsonl(inference_output_file_path, cur_iter_file, next_iter_file, add_original_data):
    data = []
    for filename in os.listdir(inference_output_file_path):
        if filename.startswith("inference") and filename.endswith(".jsonl"):
            cur_file_path = os.path.join(inference_output_file_path, filename)

            with jsonlines.open(cur_file_path) as reader:
                for line in reader.iter(skip_invalid=True):
                    data.append(line)

    filtered_data = []
    for d in data:
        category = extract_category(d["item_id"])
        threshold = threshold_list[task_list.index(category)]
        if d["reward"] > threshold:
            filtered_data.append(d)

    # filter duplicate items with same item_id
    unique_item_ids = set()
    unique_filtered_data = []
    # count category
    category_count = {}

    for entry in filtered_data:
        item_id = entry.get("item_id")
        # remove duplicate item_id
        if item_id not in unique_item_ids:
            unique_item_ids.add(item_id)
            unique_filtered_data.append(entry)
        category = extract_category(item_id)
        if category in category_count:
            category_count[category] += 1
        else:
            category_count[category] = 1

    for category, count in category_count.items():
        print(f"{category}: {count}")

    print(len(unique_filtered_data))
    # append original data
    if add_original_data:
        with open(cur_iter_file, "r") as f:
            unique_filtered_data += json.load(f)

    print(len(unique_filtered_data))

    with open(next_iter_file, "w", encoding="utf-8") as f:
        json.dump(unique_filtered_data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL file based on reward threshold.")

    parser.add_argument("--inference_output_file_path", type=str, help="current iter inference file path")
    parser.add_argument("--cur_iter_file", type=str, help="current iter train file")
    parser.add_argument("--next_iter_file", type=str, help="next iter train file")
    parser.add_argument("--add_original_data", type=bool, default=False, help="Add original data")
    parser.add_argument(
        "--inference_output_file", type=str, default="inference.jsonl", help="current iter inference file"
    )
    args = parser.parse_args()

    print("add original data", args.add_original_data)

    filter_jsonl(
        args.inference_output_file_path,
        args.cur_iter_file,
        args.next_iter_file,
        args.add_original_data,
    )


if __name__ == "__main__":
    main()
