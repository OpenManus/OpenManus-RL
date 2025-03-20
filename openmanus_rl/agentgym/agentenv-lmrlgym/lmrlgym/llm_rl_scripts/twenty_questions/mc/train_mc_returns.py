from typing import Optional
import tyro
import nltk
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import jsonl_stream, convert_path, load_mesh, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask, MapIterable, jsonl_stream, FileOpenIterable
import os
import optax
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
from LLM_RL.algorithms.mc_returns.base_interface import mc_loss
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import Text, text_env_eval, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain, text_history_to_str
from LLM_RL.algorithms.value_rl_base.gpt2.interface import GPT2ValuePolicy
from LLM_RL.heads.linear_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.linear_head import LinearHeadConfig
from JaxSeq.shard_model import copy_sharded_pytree
from functools import partial
from JaxSeq.logs import log, pull_logs
from LLM_RL.algorithms.mc_returns.train import train_loop, eval_loss
from LLM_RL.algorithms.mc_returns.data import MCData, MCIterableDataset
from LLM_RL.algorithms.mc_returns.gpt2.interface import GPT2MCTrain, GPT2MCInference

from llm_rl_scripts.twenty_questions.env.env import TwentyQuestionsPolicyEnvironment
from llm_rl_scripts.twenty_questions.env.oracle import T5Oracle
from llm_rl_scripts.twenty_questions.env.oracle import T5ModelLoadMode as T5OracleModelLoadMode
from llm_rl_scripts.twenty_questions.env.data import create_trajectories_from_conversations, get_default_word_list, create_conversation_from_history 
import json

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 
    eval_data_path: str, 
    oracle_model_path: str,

    /,  # Mark the end of positional arguments.

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
    init_lr: float=0.0001, 
    end_lr: float=0.0001, 
    lr: float=0.0001, 
    lr_warmup_steps: int=1000, 
    lr_decay_steps: int=1001, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    resid_pdrop: float=0.05, 
    attn_pdrop: float=0.05, 
    embd_pdrop: float=0.05, 

    train_bsize: int=4, 
    grad_accum_steps: Optional[int]=32, 

    train_bsize: int=32, 
    grad_accum_steps: Optional[int]=32, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 

    bf16_activations: bool=False, 

    max_length: int=1024, 

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

    policy_n_rollouts: int=32, 
    policy_bsize: int=1, 
    policy_max_input_length: int=256, 
    policy_max_output_length: int=256, 
    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None, 
    policy_bsize: int=2, 
    policy_n_rollouts: int=32, 

    eval_loss_bsize: int=32, 
    eval_loss_batches: Optional[int]=None, 
    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    input_args = dict(locals())
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")
    
    # load data
    with open(convert_path(train_data_path), 'r') as f:
        raw_train = json.load(f)
    with open(convert_path(eval_data_path), 'r') as f:
        raw_eval = json.load(f)

    train_text_trajectories = create_trajectories_from_conversations(raw_train)
    eval_text_trajectories = create_trajectories_from_conversations(raw_eval)

    def mc_data_generator(trajectories):
        for trajectory in trajectories:
            trajectory_chain = TextTrajectoryChain(text_trajectory=trajectory, 
                                                   next=None,)
            token_trajectory = TokenTrajectoryChain.from_text_trajectory_chain(trajectory_chain, tokenizer)
            yield MCData.from_token_trajectory_chain(token_trajectory, gamma=gamma)

    train_dataset = MCIterableDataset.from_mc_data_iterable(
        mc_data_generator(train_text_trajectories),
        tokenizer, 
        BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_length, 
        ), 
    )

    eval_dataset = MCIterableDataset.from_mc_data_iterable(
        mc_data_generator(eval_text_trajectories),
        tokenizer, 
        BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_length, 
        ), 
    )

    def policy_optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
    
    def value_head_optim_getter(params: PyTree):
        mask = get_weight_decay_mask(("bias",))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )

    model_prng_key = jax.random.PRNGKey(3)
    base_train_state, base_model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=policy_optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    base_model.config.gradient_checkpointing = gradient_checkpointing
    base_model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    pi_beta_params = copy_sharded_pytree(
        model=base_model, 
        pytree=base_train_state.params, 
    )

    q_prng_key = jax.random.PRNGKey(4)
    q_head_train_state, q_head = load_head_train_state_from_config(
        model_config=LinearHeadConfig(
            input_dim=base_model.config.n_embd, 
            output_dim=base_model.config.vocab_size, 
            use_bias=True, 
            initializer_range=0.0, 
            bias_init=-4.4, 
        ), 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        prng_key=q_prng_key, 
        pad_to_output_dim=None, 
        params_dtype=jnp.float32, 
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    loss_fn = partial(mc_loss, cql_weight=cql_weight)

    train = GPT2MCTrain.load_train(
        base_train_state=base_train_state, 
        q_head_train_state=q_head_train_state, 
        base_model=base_model, 
        q_head_model=q_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn, 
        detach_q=detach_q, 
    )

    inference = GPT2MCInference.load_inference(
        pi_beta_params=pi_beta_params, 
        base_params=base_train_state.params, 
        q_head_params=q_head_train_state.params, 
        pi_beta_model=base_model, 
        base_model=base_model, 
        q_head_model=q_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn, 
        beta=beta, 
        dp_shard_logits=True, 
    )
    
    model_prng_key = jax.random.PRNGKey(2)
    policy_prng, oracle_prng = jax.random.split(model_prng_key)

    env = TwentyQuestionsPolicyEnvironment(
        oracle=T5Oracle.load_oracle(
            mesh=mesh,
            prng_key=oracle_prng,
            model_load_mode=T5OracleModelLoadMode.PARAMS,
            model_load_path=oracle_model_path,
            use_fp16_activations=False,
            use_fp16_params=False,
            max_input_length=124,
            max_output_length=4,
        ),
        word_list=get_default_word_list(),
        max_conversation_length=20,
    )
    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )

    policy_prng = jax.random.PRNGKey(0)
    def evaluate(inference: GPT2MCInference):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        policy = GPT2ValuePolicy(
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

        loss_results = eval_loss(
            inference=inference, 
            dataset=eval_dataset, 
            prng_key=None, 
            bsize=eval_loss_bsize, 
            eval_batches=eval_loss_batches, 
        )

        interaction_raw_results, interaction_summary_results = text_env_eval(
            env=env, 
            policy=policy, 
            n_rollouts=policy_n_rollouts, 
            bsize=policy_bsize, 
        )

        for item in interaction_raw_results:
            print('='*25)
            print(text_history_to_str(item[-1].post_transition_history))
            print('='*25)

        logs = pull_logs(interaction_summary_results)
        log(logs, use_wandb and is_main_process)

        return loss_results['losses']['total_loss'], {'interaction': logs, 'loss': loss_results}
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=train, 
        inference=inference, 
        evaluator=evaluate, 
        dataset=train_dataset, 
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
