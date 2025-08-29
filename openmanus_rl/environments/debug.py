from openmanus_rl.environments.env_manager import make_envs
from types import SimpleNamespace
from openmanus_rl.environments.env_package.webshop.envs import WebshopWorker
import os
import ray
from pprint import pprint

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively"""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

config = {
    "env": {
        "env_name": "Webshop",
        "seed": 0,
        "rollout": {
            "n": 1,
        },
        "webshop": {
            "use_small": True,
            "human_goals": True,
        },
        "history_length": 4,
    },
    "data": {
        "train_batch_size": 1,
        "val_batch_size": 1,
    }
}

# Convert the config dictionary to a namespace for attribute access
config = dict_to_namespace(config)

file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')

env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }

worker = WebshopWorker.remote(config.env.seed, env_kwargs)

obs, info = ray.get(worker.reset.remote(0))

pprint(obs)
pprint(info)

a = "search[hair mask]"

obs, reward, done, info = ray.get(worker.step.remote(a))

print(obs)
print(reward)
print(done)
pprint(info)