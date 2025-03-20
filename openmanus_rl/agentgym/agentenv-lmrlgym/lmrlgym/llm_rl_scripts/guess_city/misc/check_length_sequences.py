from transformers import AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import convert_path, Padding, Truncation
from JaxSeq.data import MaskIterableDataset, BlockingStrategy
from llm_rl_scripts.guess_city.env.data import create_trajectories_from_conversations

import json
# load data
train_data_path = "gcs://rl-llm-bench-dataset-internal/guess-my-city/train.json"
with open(convert_path(train_data_path), 'r') as f:
    raw_train = json.load(f)

train_text_trajectories = create_trajectories_from_conversations(raw_train)
# eval_text_trajectories = create_trajectories_from_conversations(raw_eval)

def convert_trajectory_to_text(trajectories):
    for trajectory in trajectories:
        text_history = trajectory.text_history
        lst = []
        for text in text_history:
            lst.append(text.text)
        s = ' '.join(lst)
        yield s

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

iterator = convert_trajectory_to_text(train_text_trajectories)

#tokenize each item of list and find length
def tokenize_and_find_length(iterator):
    for item in iterator:
        tokenized = tokenizer(item, padding=True, truncation=True, return_tensors='np')
        yield tokenized

tokenized = tokenize_and_find_length(iterator)

lengths = []
over_1024 = 0
for item in tokenized:
    length = item['input_ids'].shape[1]
    print(length)
    # lengths.append(length)
    if length >= 1024:
        over_1024 += 1

print(sum(lengths)/len(lengths))
print(over_1024/len(lengths) * 100)