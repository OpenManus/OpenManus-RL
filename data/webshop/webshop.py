import json
from datasets import load_dataset

# Load the full AgentEval dataset
ds = load_dataset("AgentGym/AgentEval", split="test")

# Filter only the entries with item_id starting with "webshop_"
webshop_ds = ds.filter(lambda x: x["item_id"].startswith("webshop_"))

# Preview the result
print(webshop_ds)

output_file = "webshop_inference.json"

data = [{"item_id": x["item_id"], "conversations": []} for x in webshop_ds]

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)
