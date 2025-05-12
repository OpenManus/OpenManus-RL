from typing import Optional, Dict, Any, Tuple
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, get_dtype, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, uuid_name, jsonl_load, get_weight_decay_mask, create_path, get_enabled_save_path, MapIterable, FileOpenIterable
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Train, GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from JaxSeq.generation_eval import generate_language, compute_metrics
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import TextEnv, TextHistory, Text, interact_environment, text_env_eval, TextTrajectory, TextTrajectoryChain, TokenTrajectory, text_history_to_str
from JaxSeq.shard_model import shard_params_from_params
from flax.training.train_state import TrainState
from LLM_RL.utils import get_tensor_stats_np
from functools import partial
import numpy as np
from JaxSeq.logs import label_logs, log, pull_logs
import json
import random
from JaxSeq.utils import multihost_device_get
from JaxSeq.data import MaskIterableDataset
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env
from dataclasses import replace
from JaxSeq.models.gpt2.interface import loss_fn_mask
from JaxSeq.data import MaskIterableDataset, MaskDataset
from JaxSeq.models.gpt2.interface import GPT2TrainMask, GPT2InferenceMask
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from LLM_RL.algorithms.online_filtered_bc.train import train_loop
from IPython import embed

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=True, 
    wandb_project: Optional[str]="partially_observed_online_filtered_bc", 

    n_rounds: int=100, 
    epochs: int=1, 
    max_steps: Optional[int]=None, 
    
    lr: float=1e-4, 
    weight_decay: float=0.0, 

    train_bsize: int=32, 
    grad_accum_steps: Optional[int]=4, 
    rollout_bsize: int=32, 
    n_rollouts: int=128, 
    filter_percengage: float=0.3,

    bf16_activations: bool=False, 
    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 

    max_input_length: int=512, 
    max_output_length: int=6, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=None, 
    eval_every_epochs: Optional[int]=None, 
    eval_every_rounds: Optional[int]=10, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 

    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_every_rounds: Optional[int]=1, 
    save_at_beginning: bool=False, 
    save_at_end: bool=True, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_filtered_bc_dataset: bool=True, 
    save_bf16: bool=True, 

    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
    
    maze_name: str="double_t_maze",
    describe_function: str="describe_observation_only_walls", 
    reward_function: str="standard_reward",
):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)
        optim = optax.adamw(
            learning_rate=lr, 
            b1=0.9, 
            b2=0.95, 
            eps=1e-8, 
            weight_decay=weight_decay, 
            mask=mask, 
        )
        if grad_accum_steps is not None:
            optim = optax.MultiSteps(
                optim, 
                every_k_schedule=grad_accum_steps, 
            )
        return optim

    model_prng_key = jax.random.PRNGKey(2)
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    model.config.resid_pdrop = 0.0
    model.config.embd_pdrop = 0.0
    model.config.attn_pdrop = 0.0

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)

    policy_inference = GPT2InferenceMask.load_inference(
        params=train_state.params,
        model=model,
        tokenizer=tokenizer,
    )

    env = setup_maze_env(maze_name=maze_name, describe_function=describe_function, reward_function=reward_function, last_k=40)
    
    policy_prng = jax.random.PRNGKey(0)
    policy = GPT2PPOPolicy(
        inference=policy_inference,
        prng_key=policy_prng,
        generation_config=GenerationConfig(
            do_sample=policy_do_sample,
            num_beams=policy_num_beams,
            temperature=policy_temperature,
            top_p=policy_top_p,
            top_k=policy_top_k,
            eos_token_id=tokenizer.encode('\n')[0],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_output_length,
        ),
        blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT,
            truncation=Truncation.LEFT,
            max_length=max_input_length,
        ),
        out_str_process=lambda x: x.removesuffix('\n')+'\n',
    )

    ppo_inference = GPT2InferenceMask.load_inference(
        params=train_state.params,
        model=model, 
        tokenizer=tokenizer,
    )

    ppo_trainer = GPT2TrainMask.load_train(
        train_state=train_state,
        model=model,
        tokenizer=tokenizer,
    )

    data_round = 0
    def filtered_bc_dataset_loader(ppo_inference: GPT2InferenceMask, policy: GPT2PPOPolicy) -> MaskIterableDataset:
        nonlocal data_round
        raw_results, summary_results = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=n_rollouts, 
            bsize=rollout_bsize, 
        )
        summary_results = pull_logs(summary_results)

        mask_str_segments = []
        trajectory_rewards = []
        for raw_result in raw_results:
            if raw_result[-1].reward == 0:
                print('='*25)
                print(text_history_to_str(raw_result[-1].post_transition_history))
                print('='*25)
                print(sum([[item.reward, 0.0] for item in raw_result], [0.0]))
                print('='*25)

                mask_str_segments.append(
                    [(history_item.text, float(history_item.is_action)) for history_item in raw_result[-1].post_transition_history]
                )
                trajectory_rewards.append(
                    sum([item.reward for item in raw_result])
                )
        
        top_mask_str_segments = mask_str_segments
        
        
        if len(top_mask_str_segments) == 0:
            return None
        try: 
            filtered_bc_dataset = MaskDataset.blocked_from_str_segments_list(
                top_mask_str_segments,
                tokenizer,
                blocking_strategy=BlockingStrategy(
                    padding=Padding.RIGHT,
                    truncation=Truncation.LEFT,
                    max_length=max_input_length+max_output_length,
                )
            )
        except:
            embed()

        logs = dict(
            env_interaction=summary_results, 
        )

        logs = pull_logs(label_logs(logs, 'data_collection', {'round': data_round}))
        log(logs, use_wandb and is_main_process)

        if save_dir is not None and save_filtered_bc_dataset:
            print('saving filtered bc dataset ...')
            data_save_path = os.path.join(save_dir, 'data_saves', f'{data_round}')
            if is_main_process:
                create_path(data_save_path)
            # save ppo_dataset
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'filtered_bc_dataset.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(filtered_bc_dataset, f)
            # save text_trajectory_chains
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'top_mask_str_segments.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(top_mask_str_segments, f)
            # save raw_results
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'raw_results.pkl'), 
                enabled=is_main_process, 
            ), 'wb') as f:
                pkl.dump(raw_results, f)
            # save summary_results
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'summary_results.json'), 
                enabled=is_main_process, 
            ), 'w') as f:
                json.dump(summary_results, f)
            print('done saving ppo dataset.')
        
        data_round += 1

        return filtered_bc_dataset

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    ppo_trainer, ppo_inference, policy = train_loop(
        trainer=ppo_trainer, 
        inference=ppo_inference, 
        policy=policy, 
        load_dataset=filtered_bc_dataset_loader, 
        evaluator=None, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        n_rounds=n_rounds, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_every_rounds=eval_every_rounds, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_every_rounds=save_every_rounds, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
