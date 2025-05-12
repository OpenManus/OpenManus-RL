from typing import Optional, Any
import tempfile
import gcsfs
import tyro
from JaxSeq.utils import convert_path
import os
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModel
from JaxSeq.checkpointing import save_pytree
from JaxSeq.bucket_manager import open_with_bucket as open

def load_checkpoint_huggingface_from_bucket(model_output_path: str, model, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None):
    tmp_dir = tempfile.TemporaryDirectory()
    gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).get(model_output_path, tmp_dir.name, recursive=True)
    loaded = model.from_pretrained(tmp_dir.name)
    tmp_dir.cleanup()
    return loaded

def load_pickle_q_head(model_output_path: str, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None):
    tmp_dir = tempfile.TemporaryDirectory()
    gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).get(model_output_path, tmp_dir.name, recursive=True)
    with open(os.path.join(tmp_dir.name, 'q1_head.pkl'), 'rb') as f:
        q1_head = pkl.load(f)
    # do same from q2_head 
    with open(os.path.join(tmp_dir.name, 'q2_head.pkl'), 'rb') as f:
        q2_head = pkl.load(f)
    with open(os.path.join(tmp_dir.name, 'v_head.pkl'), 'rb') as f:
        v_head = pkl.load(f)
    tmp_dir.cleanup()
    return q1_head, q2_head, v_head 
def main(
    pi_beta_dir: str,
    value_base_dir: str, 
    base_dir: str,
    /,  # Mark the end of positional arguments.
):
    pi_beta_model = load_checkpoint_huggingface_from_bucket(
        os.path.join(convert_path(load_dir)),
        FlaxGPT2LMHeadModel,
        gcloud_project="rail-tpus",
        gcloud_token=None,
    )
    
    value_base = load_checkpoint_huggingface_from_bucket(
        os.path.join(convert_path(value_base_dir), 'value_base'),
        FlaxGPT2LMHeadModel,
        gcloud_project="rail-tpus",
        gcloud_token=None,
    )
    save_pytree(model.params, os.path.join(convert_path(value_base_dir), 'params.msgpack'))
    with open(os.path.join(convert_path(load_dir), 'LLM_RL', 'config.json'), 'w') as f:
        f.write(model.config.to_json_string())

if __name__ == "__main__":
    tyro.cli(main)

  