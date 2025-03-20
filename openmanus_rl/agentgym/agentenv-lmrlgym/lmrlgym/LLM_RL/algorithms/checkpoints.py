from typing import Any, Optional
from jaxtyping import PyTree
import os
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import tempfile
import gcsfs

def save_checkpoint_huggingface(model_output_path: str, model: FlaxPreTrainedModel, 
                         params: PyTree, gcloud_project: Optional[str]=None, 
                         gcloud_token: Optional[Any]=None) -> None:
    if model_output_path.startswith('gcs://'):
        model_output_path = model_output_path[len('gcs://'):]
        # save to tmp_dir
        tmp_dir = tempfile.TemporaryDirectory()
        model.save_pretrained(
            tmp_dir.name, 
            params=params, 
        )
        # upload to gcloud bucket
        gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).put(tmp_dir.name, model_output_path, recursive=True)
        # delete temp_dir
        tmp_dir.cleanup()
    else:
        model.save_pretrained(
            model_output_path, 
            params=params, 
        )

def delete_checkpoint(checkpoint_path: str, gcloud_project: Optional[str]=None, gcloud_token: Optional[Any]=None) -> None:
    if checkpoint_path.startswith('gcs://'):
        checkpoint_path = checkpoint_path[len('gcs://'):]
        gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).rm(checkpoint_path, recursive=True)
    else:
        os.system('rm -rf %s' % (checkpoint_path))