import contextlib
from typing import Any, Dict, List, Optional
import jax
from JaxSeq.models.gpt2.interface import GPT2TrainMask, GPT2InferenceMask
from JaxSeq.optimizers import GPT3Optimizer
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save, MapIterable, BlockingStrategy, Padding, Truncation
from JaxSeq.utils import get_weight_decay_mask
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode, load_params
import numpy as np
from jax.experimental.maps import Mesh
import optax
import tyro
from functools import partial
from LLM_RL.environment import text_env_eval
from llm_rl_scripts.car_dealer.env.policies import text_history_to_str
from LLM_RL.environment import TokenHistory
from JaxSeq.utils import convert_path
import os
import pickle as pkl
import json
from LLM_RL.algorithms.bc.core import bc_loss, load_bc_inference, load_bc_trainer
import pickle as pkl
from LLM_RL.algorithms.bc.data import BCDataset, filter_generator, filter_items
from LLM_RL.algorithms.bc.basic_train_loop import train_loop, eval_loop
from llm_rl_scripts.car_dealer.env.data import create_trajectories_from_conversations, Role
from llm_rl_scripts.car_dealer.env.env import BatchedCarDealerPolicyEnvironment
from llm_rl_scripts.car_dealer.env.buyer import BatchedGPT2BuyerPolicy
import tree
from transformers import AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import GenerationConfig
import jax.numpy as jnp
from jaxtyping import PyTree
import re

def main(
    model_load_mode: ModelLoadMode, 
    model_name: str, 
    buyer_model_path: str,

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str], 
    role: Role=Role.SELLER, 
    outputs_path: str="outputs/", 

    top_p: Optional[float]=None, 

    checkpoint_path: Optional[str]=None, 
    checkpoint_is_sharded: bool=True, 

    data_path: Optional[str]='data/car_dealer', 

    use_wandb: bool=False, 
    wandb_project: Optional[str]='car-dealer-bc', 

    do_pjit: bool=True, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 


    epochs: int=2, 
    max_steps: Optional[int]=None, 
    eval_batches: Optional[int]=None, 
    
    use_adafactor: bool=False,

    init_lr: float=0.0,
    end_lr: float=0.0001,
    lr: float=0.0001,
    lr_warmup_steps: int=1,
    lr_decay_steps: int=2, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True,


    weight_decay: float=0.0, 

    train_bsize: int=32, 
    grad_accum_steps: int=4, 

    policy_bsize: int=1, 

    gradient_checkpoint: bool=True, 

    max_sequence_length: int=1024, 

    policy_do_sample: bool=False,
    policy_num_beams: int=1,
    policy_temperature: float=1.0,
    policy_top_p: float=1.0,
    policy_top_k: int=0,
    policy_max_output_length: int=1024,
    policy_max_input_length: int=1024,

    bf16_activations: bool=False,

    policy_n_rollouts: int=1, 

    eval_every_steps: Optional[int]=256, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 
    
    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_bf16: bool=True, 

    force_pad_embeddings: bool=False, 

    log_every: Optional[int]=None, 
    num_logs_per_epoch: int=10,
    eval_every: Optional[int]=None,
    num_evals_per_epoch: int=5, 
    save_every: Optional[int]=None, 
    num_saves_per_epoch: int=1,
    save_best_also: bool=False,
    save_last: bool=False,

    inference_bsize: int=32, 
    seed: int=0,
    should_restore_loop_state: bool=False,

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 
):
    if use_adafactor:
        assert weight_decay == 0.0, 'no weight decay with adafactor'
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    input_args = locals().copy()
    print(input_args)

    is_main_process = jax.process_index() == 0
    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    
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
        train_text_trajectories.extend(create_trajectories_from_conversations(convos, role))
    for personality, convos in raw_eval.items():
        eval_text_trajectories.extend(create_trajectories_from_conversations(convos, role))

    print(f"Initial dataset sizes: train: {len(train_text_trajectories)}, eval: {len(eval_text_trajectories)}")

    if top_p is not None:
        train_text_trajectories = filter_items(lambda x: sum(x.reward), train_text_trajectories, take_top=top_p, threshold=None)
        eval_text_trajectories = filter_items(lambda x: sum(x.reward), eval_text_trajectories, take_top=top_p, threshold=None)

    train_text_histories = [trajectory.text_history for trajectory in train_text_trajectories]
    eval_text_histories = [trajectory.text_history  for trajectory in eval_text_trajectories]

    train_token_histories = [TokenHistory.from_text_history(text_history, tokenizer) for text_history in train_text_histories]
    eval_token_histories = [TokenHistory.from_text_history(text_history, tokenizer) for text_history in eval_text_histories]
    
    train_token_histories = [token_history for token_history in train_token_histories if token_history.tokens.shape[0] <= max_sequence_length]
    eval_token_histories = [token_history for token_history in eval_token_histories if token_history.tokens.shape[0] <= max_sequence_length]

    print(f"Final dataset sizes: train: {len(train_token_histories)}, eval: {len(eval_token_histories)}")

    train_data = BCDataset(
        token_histories=train_token_histories, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )

    eval_data = BCDataset(
        token_histories=eval_token_histories, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )
    
    if checkpoint_is_sharded and checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(checkpoint_path.strip('/'))
        checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)
    
    if model_name == 'gpt2-xl' or model_name == 'gpt2-medium':
        print("loading model")
        train_state, model = load_train_state(
            model_load_mode=model_load_mode, 
            model_load_path=convert_path(model_name) if model_load_mode != ModelLoadMode.HF else model_name, 
            model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
            optim_getter=optim_getter, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=model_prng_key, 
            force_pad_embeddings=force_pad_embeddings, 
            params_dtype=jnp.float32, 
        )
    else:
        raise NotImplementedError

    def optim_getter(params: PyTree):
            mask = get_weight_decay_mask((
                "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
                "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
                re.escape("['ln_f']['bias']"), 
                re.escape("['ln_f']['scale']"), 
                "bias", 
            ))(params)
            
            optimizer_config = GPT3Optimizer(
                init_lr=init_lr, 
                end_lr=end_lr, 
                lr=lr, 
                lr_warmup_steps=lr_warmup_steps, 
                lr_decay_steps=lr_decay_steps, 
                weight_decay=weight_decay, 
                bf16_momentum=bf16_momentum, 
                multiply_by_parameter_scale=multiply_by_parameter_scale, 
            )

            optim, _ = optimizer_config.get_optim(mask)

            if grad_accum_steps is not None:
                return optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
            return optim

    model_prng_key = jax.random.PRNGKey(2)
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_name) if model_load_mode != ModelLoadMode.HF else model_name, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    # model.config.gradient_checkpointing = gradient_checkpointing
    # model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    # model.config.resid_pdrop = resid_pdrop
    # model.config.embd_pdrop = embd_pdrop
    # model.config.attn_pdrop = attn_pdrop

    # loop_state = dict()
    # if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
    #                                                       ModelLoadMode.TRAIN_STATE_PARAMS, 
    #                                                       ModelLoadMode.PARAMS}):
    #     with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
    #         loop_state = pkl.load(f)

    print("loading trainer and inference")    
    trainer = GPT2TrainMask.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = GPT2InferenceMask.load_inference(
        params=train_state.params, 
        model=model, 
        tokenizer=tokenizer, 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )
    
    policy_prng = jax.random.PRNGKey(0)
    print("load environment")
    prng_key = jax.random.PRNGKey(3)
    prng_key, buyer_inference_prng, buyer_policy_prng = jax.random.split(prng_key, 3)
    buyer_model_mode = ModelLoadMode.PARAMS
    buyer_params, buyer_model = load_params(
        model_load_mode=buyer_model_mode, 
        model_load_path=convert_path(buyer_model_path) if model_load_mode != ModelLoadMode.HF else buyer_model_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=buyer_inference_prng, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )

    env = BatchedCarDealerPolicyEnvironment(
        buyer=buyer_model, 
        bsize=8,
    )

    def evaluator(inference: GPT2InferenceMask):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        policy = GPT2PPOPolicy(
            inference=inference, 
            prng_key=new_key, 
            generation_config=GenerationConfig(
                do_sample=policy_do_sample, 
                num_beams=policy_num_beams, 
                temperature=policy_temperature, 
                top_p=policy_top_p, 
                top_k=policy_top_k, 
                eos_token_id=tokenizer.encode('\n')[0], 
                pad_token_id=tokenizer.pad_token_id, 
                max_new_tokens=policy_max_output_length, 
            ), 
            blocking_strategy=BlockingStrategy(
                padding=Padding.LEFT, 
                truncation=Truncation.LEFT, 
                max_length=policy_max_input_length, 
            ), 
            out_str_process=lambda x: x.removesuffix('\n')+'\n', 
        )

        data_results = eval_loop(
            inference=inference, 
            dataset=eval_data, 
            rng=jax.random.PRNGKey(1), 
            bsize=32, 
            prefetch_batches=None, 
            eval_batches=16, 
        )
        
        interation_raw_results, interaction_summary_results = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=policy_n_rollouts, 
            bsize=policy_bsize, 
        )

        for item in interation_raw_results:
            print('='*25)
            print(text_history_to_str(item[-1].post_transition_history))
            print('='*25)

        return data_results['loss'], {'loss_metrics': data_results, 'generation_metrics': interaction_summary_results}
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=trainer, 
        inference=inference, 
        evaluator=evaluator, 
        dataset=train_data, 
        rng=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every=eval_every_steps,  
        eval_at_beginning=eval_at_beginning,  
        save_every=save_every_steps,  
        save_at_end=save_at_end, 
        save_best=save_best,  
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None,  
    )

if __name__ == "__main__":
    tyro.cli(main)