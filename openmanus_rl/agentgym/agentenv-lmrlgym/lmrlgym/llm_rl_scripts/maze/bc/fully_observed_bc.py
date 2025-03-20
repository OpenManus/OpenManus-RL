from typing import Optional, List
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save, MapIterable, BlockingStrategy, Padding, Truncation
import jax
import jax.numpy as jnp
from JaxSeq.utils import get_weight_decay_mask
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Train, GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqDataset
from JaxSeq.train import eval_loss, train_loop
from jaxtyping import PyTree
import re
from JaxSeq.optimizers import GPT3Optimizer
import json
import numpy as np
from transformers import AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from LLM_RL.algorithms.ppo.reranker_policy import ReRankerSamplePolicy
from LLM_RL.algorithms.ppo.score_fn import build_bc_score_fn
import random
from LLM_RL.environment import text_env_eval
from llm_rl_scripts.maze.env.env import maze_proposal_function
from llm_rl_scripts.maze.env.maze_utils import pick_start_position, setup_maze_env
from llm_rl_scripts.maze.env.mazes import double_t_maze
from transformers.generation import GenerationConfig

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str,

    /,  # Mark the end of positional arguments.
    eval_frac: float=0.1,
    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    epochs: int=1, 
    max_steps: Optional[int]=None, 

    weight_decay: float=0.001, 
    init_lr: float=0.0, 
    end_lr: float=0.0001, 
    lr: float=0.0001, 
    lr_warmup_steps: int=1, 
    lr_decay_steps: int=2, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    train_bsize: int=128, 
    grad_accum_steps: Optional[int]=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 
    bf16_activations: bool=False, 

    max_input_length: int=256, 
    max_output_length: int=8, 

    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None,

    log_every: int=256, 
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

    eval_loss_bsize: int=32, 
    eval_loss_batches: Optional[int]=None, 
    generation_bsize: int=4, 
    generation_batches: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
    traj_max_length:int=40
):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")
    
    with open(convert_path(train_data_path), "r") as f:
        all_items = list(f)
    
    def str_lst(items: List[str]):
        lst = []
        for item in items:
            obj = json.loads(item)
            for state_action_pair in obj:
                lst.append((state_action_pair["state"], state_action_pair["action"]))
        return lst
    
    # create splits
    data_lst = str_lst(all_items)
    random.seed(0)
    random.shuffle(data_lst)
    train_lst = data_lst[:int(len(data_lst)*eval_frac)]
    eval_lst = data_lst[int(len(data_lst)*eval_frac):]
    
    
    train_data = Seq2SeqDataset.from_str_list(
        train_lst, 
        tokenizer=tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length, 
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length
        ), 
    )

    eval_data = Seq2SeqDataset.from_str_list(
        eval_lst, 
        tokenizer=tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length, 
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length, 
        ), 
    )

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_\d+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_\d+'\]", re.escape("['scale']")]), 
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

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    trainer = GPT2Train.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = GPT2Inference.load_inference(
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
    
    maze_name = "double_t_maze"
    describe_function = "describe_observation_only_walls"
    reward_function = "standard_reward"
    
    maze = double_t_maze()
    
    env = setup_maze_env(maze_name=maze_name, describe_function=describe_function, reward_function=reward_function, last_k=1)
    start_position = pick_start_position(maze_name=maze_name)
    possible_positions = list(zip(*np.where(maze==0)))
    possible_positions.remove((8, 6))
    
    def evaluator(inference: GPT2Inference):
        data_results = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=jax.random.PRNGKey(1), 
            bsize=4, 
            eval_batches=64, 
        )
        
        results = {}
        for position in possible_positions:
            position = tuple(position)
            _, results[str(position)] = text_env_eval(
                env=env, 
                policy = GPT2PPOPolicy(
                    inference=inference, 
                    prng_key=jax.random.PRNGKey(0), 
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
                ), 
                n_rollouts=1, 
                verbose=True, 
                # save_path=None, 
                # seed=1, 
                env_options={"init_position": position},
                # save_config=None, 
            )
        # avg_position_results = results["avg_reward"]
        #TODO: accuracy metric
        return data_results['loss'], {'data': data_results, 'sample_env': results}
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=trainer, 
        inference=inference, 
        evaluator=evaluator, 
        dataset=train_data, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
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
