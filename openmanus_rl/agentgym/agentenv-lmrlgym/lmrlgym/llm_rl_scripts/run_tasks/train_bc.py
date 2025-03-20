from functools import partial
from typing import Optional
from IPython import embed
import tyro
from LLM_RL.algorithms.bc.data import BCDataset, BCIterableDataset
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save, MapIterable, BlockingStrategy, Padding, Truncation
import jax
import jax.numpy as jnp
from JaxSeq.utils import get_weight_decay_mask
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Train, GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqIterableDataset
from JaxSeq.train import eval_loss, train_loop
from jaxtyping import PyTree
import re
from JaxSeq.optimizers import GPT3Optimizer
from transformers.generation import GenerationConfig
import json
from transformers import AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open
from LLM_RL.algorithms.bc.interface import bc_loss, load_bc_inference, load_bc_trainer
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from LLM_RL.environment import TokenHistory
from llm_rl_scripts.chess.env.env import preprocess_move, preprocess_state, text_env_eval_chess_positions
from JaxSeq.logs import pull_logs
from LLMRL_tasks.llm_rl.data import get_data

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    task_name: str,
    train_data_path: str, 
    eval_data_path: str, 
    test_positions_path: str, # need to fix this for endgames

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=True, 
    wandb_project: Optional[str]="llm_rl_endgames", 

    epochs: int=1, 
    max_steps: Optional[int]=None, 

    weight_decay: float=0.001, 
    init_lr: float=0.0, 
    end_lr: float=0.0001, 
    lr: float=0.0001, 
    lr_warmup_steps: int=0, 
    lr_decay_steps: int=1001, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    train_bsize: int=128, 
    grad_accum_steps: Optional[int]=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 
    bf16_activations: bool=False, 

    max_input_length: int=150, # maximum possible board state length
    max_output_length: int=10, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=10240, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=True, 
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
    traj_max_length:int=40,
    
    filtered: bool=False,
    ):
    
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    # train_items = all_items[:int(len(all_items)*eval_frac)]
    # eval_items = all_items[int(len(all_items)*eval_frac):]

    train_chain_generator, eval_chain_generator = get_data(task_name, train_data_path, eval_data_path)

    # TODO: copy in code for BCIterableDataset and BCDataset
    def str_iterable(chain_generator):
        for chain in chain_generator:
            curr_trajectory = chain.text_trajectory
            text_history = curr_trajectory.text_history
            token_history = TokenHistory.from_text_history(text_history, tokenizer)
            yield token_history
    
    if epochs == 1:
        train_data = BCIterableDataset(
            token_generator=str_iterable(train_chain_generator),
            pad_token_id=tokenizer.pad_token_id,
            max_len=max_input_length + max_output_length,
        )

        eval_data = BCIterableDataset(
            token_generator=str_iterable(eval_chain_generator),
            pad_token_id=tokenizer.pad_token_id,
            max_len=max_input_length + max_output_length,
        )
    else:
        train_data = BCDataset(
            token_histories=list(str_iterable(train_chain_generator)),
            pad_token_id=tokenizer.pad_token_id,
            max_len=max_input_length + max_output_length,
        )

        eval_data = BCDataset(
            token_histories=list(str_iterable(eval_chain_generator)),
            pad_token_id=tokenizer.pad_token_id,
            max_len=max_input_length + max_output_length,
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

    loss_fn = partial(bc_loss, non_action_weight=0.0)

    trainer = GPT2Train.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn,
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
    

    with open(test_positions_path, "r") as f:
        positions = list(f)
        positions = [position.replace("\n", "").replace("\"", "") for position in positions if position != ""]
    
    def evaluator(inference: GPT2Inference):
        data_results = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=jax.random.PRNGKey(1), 
            bsize=4, 
            eval_batches=64, 
        )
        
        policy = GPT2PPOPolicy(
            inference=inference, 
            prng_key=jax.random.PRNGKey(1), 
            generation_config=GenerationConfig(
                do_sample=True, 
                num_beams=1, 
                temperature=None, 
                top_p=None, 
                top_k=None,
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
        raw_results, summary_results = text_env_eval_chess_positions(
            positions=positions,
            policy=policy,
            n_rollouts=1,
            bsize=1,
        )
        summary_results = pull_logs(summary_results)
        
        return data_results['loss'], {'data': data_results, 'interaction_env': summary_results}
    
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
