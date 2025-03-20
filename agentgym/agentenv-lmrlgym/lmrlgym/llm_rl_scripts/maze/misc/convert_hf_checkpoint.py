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

def main(
    load_dir: str, 
    /,  # Mark the end of positional arguments.
):
    model = load_checkpoint_huggingface_from_bucket(
        os.path.join(convert_path(load_dir)),
        FlaxGPT2LMHeadModel,
        gcloud_project="rail-tpus",
        gcloud_token=None,
    )
    save_pytree(model.params, os.path.join(convert_path(load_dir), 'LLM_RL', 'params.msgpack'))
    with open(os.path.join(convert_path(load_dir), 'LLM_RL', 'config.json'), 'w') as f:
        f.write(model.config.to_json_string())

if __name__ == "__main__":
    tyro.cli(main)

