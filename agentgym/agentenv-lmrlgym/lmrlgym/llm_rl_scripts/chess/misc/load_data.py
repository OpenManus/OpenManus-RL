from llm_rl_scripts.chess.env.env import FenChessHistoryEnv, preprocess_move, preprocess_state
from JaxSeq.bucket_manager import open_with_bucket as open
import json 
import random
from IPython import embed

filtered = False
data_path = "gcs://rl-llm-bench-dataset/chess/complete_background_generated/train.jsonl"
data = []
def str_iterable(data_path):
    with open(data_path, "r") as f:
        for obj in f:
            # print(obj)
            if obj is None or obj == "":
                continue
            result = json.loads(obj)
            yield {"in_text": preprocess_state(result["from_state"]), "out_text": preprocess_move(result["action"])}
            # embed()
            # # embed()
            # if (filtered and result[-1]["reward"] == 1) or not filtered:
            #     for item in result:
            #         item = {"in_text": preprocess_state(item["state"]), "out_text": preprocess_move(item["action"])}
            #         data.append(item)

# random.shuffle(data)

iterable = str_iterable(data_path)
for _ in range(10):
    print(next(iterable))