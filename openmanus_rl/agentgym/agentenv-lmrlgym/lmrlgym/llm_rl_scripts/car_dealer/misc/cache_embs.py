import contextlib
from typing import Optional, Tuple
import jax
from algorithms.jax_agent import JaxAgentPolicy, Inference
from algorithms.jax_bc.basic_train_loop import train_loop, eval_loop
from algorithms.jax_bc.core import bc_loss, load_bc_inference, load_bc_trainer
from algorithms.jax_bc.data import BCDataset, BCIterableDataset
from algorithms.jax_ilql.models import BaseModelState
from jax_models.gptj import load_gptj_model
from jax_utils.jax_shard import OptimType, shard_optim_and_params, shard_params
from environment import TextTrajectory, TextTransition
from environments.car_dealer.data import create_trajectories_from_conversations, Role
from jax_models.gpt2 import load_gpt2_model
from transformers import GPT2Tokenizer
import numpy as np
from jax.experimental.maps import Mesh
import optax
from dataclasses import dataclass
import tyro
from functools import partial
from text_env_eval import text_env_eval
from token_history import text_history_to_token_history, text_transition_to_token_transition
from utils.path import convert_path
import os
import pickle as pkl
from algorithms.jax_ilql.data import ILQLDataset, ILQLIterableDataset
from algorithms.jax_frozen_ilql.core import load_frozen_ilql_inference
from algorithms.jax_frozen_ilql.basic_cache_embedding_loop import embedding_loop
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import json
import sys
sys.setrecursionlimit(10000)

def main(
    exp_name: Optional[str], 
    model_name: str, 

    /,  # Mark the end of positional arguments.

    role: Role=Role.SELLER, 
    reward_mode: str="fancy",

    checkpoint_path: Optional[str]=None, 
    checkpoint_is_sharded: bool=True, 

    data_path: Optional[str]='data/car_dealer', 
    output_path: Optional[str]='embs/car_dealer', 

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    bsize: int=32,
    max_sequence_length: int=1024, 

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 
):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')
    
    input_args = locals().copy()
    print(input_args)

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    with open(convert_path(os.path.join(data_path, 'train.json')), 'r') as f:
        raw_train = json.load(f)
    with open(convert_path(os.path.join(data_path, 'eval.json')), 'r') as f:
        raw_eval = json.load(f)

    train_text_trajectories = []
    eval_text_trajectories = []
    for personality, convos in raw_train.items():
        train_text_trajectories.extend(create_trajectories_from_conversations(convos, role, reward_mode))
    for personality, convos in raw_eval.items():
        eval_text_trajectories.extend(create_trajectories_from_conversations(convos, role, reward_mode))

    train_token_transitions = [text_transition_to_token_transition(TextTransition(text_trajectory, None), tokenizer) for text_trajectory in train_text_trajectories]
    eval_token_transitions = [text_transition_to_token_transition(TextTransition(text_trajectory, None), tokenizer) for text_trajectory in eval_text_trajectories]

    train_token_transitions = [token_transition for token_transition in train_token_transitions if token_transition.token_trajectory.tokens.shape[0] <= max_sequence_length]
    eval_token_transitions = [token_transition for token_transition in eval_token_transitions if token_transition.token_trajectory.tokens.shape[0] <= max_sequence_length]

    print(f"Final dataset sizes: train: {len(train_token_transitions)}, eval: {len(eval_token_transitions)}")

    train_data = ILQLDataset(
        token_transitions=train_token_transitions, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )

    eval_data = ILQLDataset(
        token_transitions=eval_token_transitions, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )

    if checkpoint_is_sharded and checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
        checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)
    
    if model_name == 'gpt2-xl' or model_name == 'gpt2-medium':
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    else:
        raise NotImplementedError

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
    else:
        mesh = contextlib.nullcontext()

    # shard params and optimizer
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                          params, shard_rules, mesh)
    else:
        param_spec = None

    base_lm_state = BaseModelState(
        model=model, 
        params=params, 
        param_spec=param_spec, 
    )
    
    inference = load_frozen_ilql_inference(
        base_lm_state=base_lm_state, 
        q1_head_state=None, 
        q2_head_state=None, 
        v_head_state=None, 
        target_q1_head_state=None, 
        target_q2_head_state=None, 
        emb_cacher=None, 
        emb_dim=model.config.hidden_size, 
        tokenizer=tokenizer, 
        do_pjit=True, 
        loss_fn=None, 
        beta=0.0, 
        use_pre_ln_state=False, 
    )

    save_dir = None
    if exp_name is not None:
        save_dir = convert_path(os.path.join(output_path, exp_name))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
    
    with mesh:
        print("Starting train embedding.")
        emb_cacher = embedding_loop(
            inference=inference, 
            dataset=train_data, 
            rng=jax.random.PRNGKey(1), 
            bsize=bsize, 
            prefetch_batches=None, 
            max_batches=None, 
        )
    
    if save_dir is not None:
        with open(os.path.join(save_dir, 'train_embs.pkl'), 'wb') as f:
            pkl.dump(emb_cacher, f)
        with open(os.path.join(save_dir, 'train_token_transitions.pkl'), 'wb') as f:
            pkl.dump(train_token_transitions, f)
    
    with mesh:
        print("Starting eval embedding.")
        emb_cacher = embedding_loop(
            inference=inference, 
            dataset=eval_data, 
            rng=jax.random.PRNGKey(1), 
            bsize=bsize, 
            prefetch_batches=None, 
            max_batches=None, 
        )
    
    if save_dir is not None:
        with open(os.path.join(save_dir, 'eval_embs.pkl'), 'wb') as f:
            pkl.dump(emb_cacher, f)
        with open(os.path.join(save_dir, 'eval_token_transitions.pkl'), 'wb') as f:
            pkl.dump(eval_token_transitions, f)

if __name__ == "__main__":
    tyro.cli(main)
