import json
from JaxSeq.bucket_manager import open_with_bucket as open

data_name = "gcs://rl-llm-bench-dataset-internal/endgames/bc_train_filtered.jsonl"

with open(data_name, "r") as f:
    item = f.readline()
    obj = json.loads(item)
    print(obj)
    
data_name = "gcs://rl-llm-bench-dataset-internal/endgames/bc_val.jsonl"

with open(data_name, "r") as f:
    item = f.readline()
    obj = json.loads(item)
    print(obj)