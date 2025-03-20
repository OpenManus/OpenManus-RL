import contextlib
from typing import Any, Dict, List, Optional
import jax
from algorithms.jax_agent import Inference, JaxAgentPolicy
from algorithms.jax_bc.core import load_bc_inference
from algorithms.jax_frozen_ilql.core import load_frozen_ilql_inference
from algorithms.jax_frozen_ilql.load_frozen_ilql import load_frozen_ilql_default_full_inference
from algorithms.jax_ilql.models import BaseModelState, load_advanced_mlp4
from jax_utils.jax_shard import shard_params
from environment import Text, TextHistory, TextTransition, interact_environment
from jax_models.gpt2 import load_gpt2_model
from jax_models.t5 import load_t5_model
import numpy as np
from jax.experimental.maps import Mesh
import dcargs
from functools import partial
from utils.path import convert_path
from utils.randomness import seed_generator
import os
import pickle as pkl
import json
import random
from utils.randomness import seed_context
from environments.twenty_questions.data import asker_postproc, asker_postproc_simple, asker_postproc_filter_repeats, get_default_word_list, create_conversation_from_history, conversation_to_str
from environments.twenty_questions.env import TwentyQuestionsPolicyEnvironment
from environments.twenty_questions.oracle import load_flan_t5_xl_oracle
import tree
from transformers import AutoTokenizer
from collections import defaultdict


def load_bc_policy(
    model_name: str, 
    rng: jax.random.KeyArray, 
    mesh: Mesh,
    checkpoint_path: Optional[str]=None, 
    checkpoint_is_sharded: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    do_pjit: bool=True, 
    condition_str: Optional[str]=None, 
    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
    simple_postproc: bool=False,
):
    if mesh is None:
        mesh = contextlib.nullcontext()
        data_p_shape = 1
    else:
        data_p_shape = mesh.devices.shape[0]

    if checkpoint_is_sharded and checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
        checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

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
    
    # shard params and optimizer
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                          params, shard_rules, mesh)
    else:
        param_spec = None
    
    inference = load_bc_inference(
        model=model, 
        params=params, 
        param_spec=param_spec, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=None, 
    )

    policy = JaxAgentPolicy(
        inference=inference, 
        tokenizer=tokenizer, 
        rng=rng, 
        max_input_length=max_sequence_length-max_new_tokens, 
        condition_str=condition_str, 
        postproc_f=asker_postproc_filter_repeats, 
        data_parallel_mesh_shape=data_p_shape, 
        do_sample=do_sample, 
        num_beams=1, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.encode('\n')[0], 
        max_new_tokens=max_new_tokens, 
    )

    return policy


def load_frozen_policy(
    model_name: str, 
    rng: jax.random.KeyArray, 
    mesh: Mesh,
    lm_checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None, 
    value_checkpoint_idx: Optional[int]=None, 
    checkpoint_is_sharded: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    do_pjit: bool=True, 
    condition_str: Optional[str]=None, 
    beta: float=1.0, 
    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
):
    print("Loading frozen policy.")
    if mesh is None:
        mesh = contextlib.nullcontext()
        data_p_shape = 1
    else:
        data_p_shape = mesh.devices.shape[0]

    if checkpoint_is_sharded and lm_checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(lm_checkpoint_path.strip('/'))
        lm_checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)

    if checkpoint_is_sharded and value_checkpoint_path is not None:
        value_checkpoint_path = os.path.join(value_checkpoint_path, 'shard_%d' % (jax.process_index()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    if model_name == 'gpt2-xl':
        print(f"Loading gpt2-xl lm model from {lm_checkpoint_path}")
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=lm_checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=False, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
    else:
        raise NotImplementedError
    
    model_config = model.config
    
    q_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=True, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    v_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=False, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    
    value_path_suffix = f'_{value_checkpoint_idx}' if value_checkpoint_idx is not None else ''
    rng, ilql_state_rng = jax.random.split(rng)
    print(f"Loading value model from {value_checkpoint_path} with suffix {value_path_suffix}.")
    ilql_state = load_frozen_ilql_default_full_inference(
        q_load_head_fn=q_load_head_fn, 
        q_head_rng_keys=frozenset(['dropout']), 
        v_load_head_fn=v_load_head_fn, 
        v_head_rng_keys=frozenset(['dropout']), 
        emb_dim=model_config.hidden_size, 
        vocab_size=model_config.vocab_size, 
        rng=ilql_state_rng, 
        mesh=mesh, 
        do_pjit=do_pjit, 
        q1_checkpoint_path=os.path.join(value_checkpoint_path, f'q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        q2_checkpoint_path=os.path.join(value_checkpoint_path, f'q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        v_checkpoint_path=os.path.join(value_checkpoint_path, f'v_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q1_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q2_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
    )
    
    # shard params
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), params, shard_rules, mesh)
    else:
        param_spec = None

    base_lm_state = BaseModelState(
        model=model, 
        params=params, 
        param_spec=param_spec, 
    )

    inference = load_frozen_ilql_inference(
        base_lm_state=base_lm_state, 
        q1_head_state=ilql_state.q1_head_model_state, 
        q2_head_state=ilql_state.q2_head_model_state, 
        v_head_state=ilql_state.v_head_model_state, 
        target_q1_head_state=ilql_state.target_q1_head_model_state, 
        target_q2_head_state=ilql_state.target_q2_head_model_state, 
        emb_cacher=None, 
        emb_dim=model_config.hidden_size, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=None, 
        beta=beta, 
        use_pre_ln_state=False, 
    )

    rng, policy_rng = jax.random.split(rng)
    policy = JaxAgentPolicy(
        inference=inference, 
        tokenizer=tokenizer, 
        rng=policy_rng, 
        max_input_length=max_sequence_length-max_new_tokens, 
        condition_str=condition_str, 
        postproc_f=asker_postproc_filter_repeats, 
        data_parallel_mesh_shape=data_p_shape, 
        do_sample=do_sample, 
        num_beams=1, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.encode('\n')[0], 
        max_new_tokens=max_new_tokens, 
    )

    return policy


def main(
    exp_name: str, 
    model_name: str, 
    algorithm_name: str,

    /,  # Mark the end of positional arguments.

    checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None,
    value_checkpoint_idx: Optional[str]=None,
    checkpoint_is_sharded: bool=True, 

    output_path: Optional[str]='evals/twenty_questions', 

    max_sequence_length: int=512,
    max_new_tokens: int=80,
    do_sample: bool=False,
    beta: float=1.0,

    env_deterministic: bool=False,
    num_samples: int=158,
    save_every: int=100,
    seed: int=0,

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 

    verbose: bool=False,
):
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    input_args = locals().copy()
    print(input_args)

    from utils.gcs_manager import open_pp as open
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
    else:
        mesh = contextlib.nullcontext()

    # rng
    rng = jax.random.PRNGKey(seed)

    # load guesser model
    if algorithm_name == "bc":
        rng, policy_rng = jax.random.split(rng)
        policy = load_bc_policy(
            model_name=model_name, 
            rng=policy_rng, 
            mesh=mesh,
            checkpoint_path=checkpoint_path, 
            checkpoint_is_sharded=checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            condition_str=None, 
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    elif algorithm_name == "frozen":
        rng, policy_rng = jax.random.split(rng)
        policy = load_frozen_policy(
            model_name=model_name, 
            rng=policy_rng, 
            mesh=mesh,
            lm_checkpoint_path=checkpoint_path,
            value_checkpoint_path=value_checkpoint_path,
            value_checkpoint_idx=value_checkpoint_idx,
            checkpoint_is_sharded=checkpoint_is_sharded, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
            do_pjit=do_pjit, 
            condition_str=None, 
            beta=beta,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    else:
        raise NotImplementedError

    # check output directory
    save_dir = None
    if output_path is not None:
        save_dir = convert_path(os.path.join(output_path, exp_name))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            print(f"Output directory {save_dir} does not exist. Making directory...")
            os.makedirs(save_dir)
        
        # copy script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
            pkl.dump(input_args, f)

    # create environment
    rng, oracle_rng = jax.random.split(rng)
    env = TwentyQuestionsPolicyEnvironment(
        oracle=load_flan_t5_xl_oracle(
            mesh=mesh,
            rng=oracle_rng,
            model_name="google/flan-t5-xl",
            checkpoint_path="gcs://rail-tpus-charles-3/JaxSeq/outputs/twenty_questions/flan-t5-xl_oracle_lr1e-5_test1/model_2.pkl",
            max_input_length=68, #124
            max_output_length=4,
            do_pjit=do_pjit,
            gcloud_project=gcloud_project,
            gcloud_token=gcloud_token,
        ),
        word_list=get_default_word_list(),
        max_conversation_length=20,
    )

    # If env is deterministic and policy is deterministic, then just loop through all words once.
    if env_deterministic and not do_sample:
        num_samples = len(env.word_list)
        print(f"Both environment and policy are deterministic, so changed num_samples to {num_samples}")

    # Do rollouts
    with mesh:
        random_seed = seed_generator(seed)
        
        rewards = []
        corrects = []
        rewards_per_word = defaultdict(list)
        corrects_per_word = defaultdict(list)
        conversations = []

        def save_once():
            with open(convert_path(os.path.join(save_dir, 'conversations.json')), 'w') as f:
                json.dump(conversations, f, indent=4)

            avg_rewards_per_word = {word: sum(word_r) / len(word_r) for word, word_r in rewards_per_word.items()}
            avg_corrects_per_word = {word: sum(word_c) / len(word_c) for word, word_c in corrects_per_word.items()}
            num_samples_per_word = {word: len(word_c) for word, word_c in corrects_per_word.items()}

            stats = {
                "avg_rewards": sum(rewards) / len(rewards),
                "avg_corrects": sum(corrects) / len(corrects),
                "avg_rewards_per_word": avg_rewards_per_word,
                "avg_corrects_per_word": avg_corrects_per_word,
                "num_samples_per_word": num_samples_per_word,
                "rewards": rewards,
                "rewards_per_word": rewards_per_word,
                "corrects_per_word": corrects_per_word,
            }
            with open(convert_path(os.path.join(save_dir, 'stats.json')), 'w') as f:
                json.dump(stats, f, indent=2)

            if verbose:
                print(f"saved to {save_dir}")
                print(f"avg_rewards: {stats['avg_rewards']}")
                print(f"avg_corrects: {stats['avg_corrects']}")

            return stats

        for i in range(num_samples):
            transitions = interact_environment(env, policy, env_seed=next(random_seed), env_options={"deterministic": env_deterministic})

            _, _, final_text_history, _, _ = transitions[-1]
            word_var = env.curr_word
            conversation = create_conversation_from_history(word_var, final_text_history, max_conversation_len=20)
            
            reward = sum([r for _, _, _, r, _ in transitions])

            conversations.append(conversation)
            rewards.append(reward)
            corrects.append(conversation["correct"])
            rewards_per_word[word_var[0]].append(reward)
            corrects_per_word[word_var[0]].append(conversation["correct"])

            if verbose:
                print('='*25)
                print(conversation_to_str(conversation))
                print('='*25)
                print('reward:', reward)
                print('='*25)
                print()
            
            if save_dir is not None and (i + 1) % save_every == 0:
                save_once()
        
        final_stats = save_once()
        if verbose:
            print(f"avg_rewards_per_word: {final_stats['avg_rewards_per_word']}")
            print(f"avg_corrects_per_word: {final_stats['avg_corrects_per_word']}")
            print(f"num_samples_per_word: {final_stats['num_samples_per_word']}")


if __name__ == "__main__":
    dcargs.cli(main)