import json
from JaxSeq.bucket_manager import open_with_bucket as open
import random

data_name = "gcs://rl-llm-bench-dataset-internal/endgames/val.jsonl"

data = []

with open(data_name, "r") as f:
    for item in f:
        obj = json.loads(item)
        if obj[-1]["reward"] == 1:
            for pair in obj:
                data.append({"from_state": pair["state"], "action": pair["action"]})

print(data[0:10])
random.shuffle(data)
with open("gcs://rl-llm-bench-dataset-internal/endgames/bc_val_filtered.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")