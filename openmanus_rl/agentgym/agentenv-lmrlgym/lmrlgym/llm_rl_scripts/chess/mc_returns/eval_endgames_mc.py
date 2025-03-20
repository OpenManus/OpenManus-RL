from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import convert_path, load_mesh, create_path
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation
import os
from JaxSeq.models.gpt2.interface import GPT2InferenceMask
from JaxSeq.models.gpt2.load import ModelLoadMode, load_params
import pickle as pkl
import json
from transformers.generation import GenerationConfig
from LLM_RL.environment import text_env_eval
from llm_rl_scripts.chess.env.env import text_env_eval_chess_positions
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env, maze_solver
from collections import defaultdict
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from LLM_RL.environment import Text
from LLM_RL.algorithms.value_rl_base.gpt2.interface import GPT2ValuePolicy, GPT2ValueRLInference
from LLM_RL.heads.mlp_head import load_params as load_head_params
from LLM_RL.algorithms.mc_returns.score_fn import build_mc_score_fn

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str,
    pi_beta_load_mode: ModelLoadMode,
    pi_beta_load_path: str,
    test_positions_path: str, 

    /,  # Mark the end of positional arguments.

    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    bf16_activations: bool=False, 

    policy_n_rollouts: int=32, 
    policy_bsize: int=1, 
    policy_max_input_length: int=256, 
    policy_max_output_length: int=256, 
    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None,
    policy_beta: float=16.0,

    do_accuracy_eval: bool=True,
    do_reward_eval: bool=True,
    use_reranker_for_reward_eval: bool=False,

    force_pad_embeddings: bool=False,
):
    assert model_load_mode != ModelLoadMode.HF
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")


    pi_beta_prng_key = jax.random.PRNGKey(0)
    pi_beta_params, _ = load_params(
        model_load_mode=pi_beta_load_mode, 
        model_load_path=convert_path(pi_beta_load_path) if pi_beta_load_mode != ModelLoadMode.HF else pi_beta_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=pi_beta_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )

    base_prng_key = jax.random.PRNGKey(0)
    base_params, base_model = load_params(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(os.path.join(model_load_path, 'base')), 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=base_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )

    q_head_params, q_head = load_head_params(
        model_load_mode=model_load_mode.value,
        model_load_path=convert_path(os.path.join(model_load_path, 'q_head')),
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32,
        mesh=mesh,
        prng_key=jax.random.PRNGKey(0),
        pad_to_output_dim=None,
        params_dtype=jnp.float32,
    )

    inference = GPT2ValueRLInference.load_inference(
        pi_beta_params=pi_beta_params, 
        base_params=base_params, 
        q1_head_params=q_head_params, 
        q2_head_params=None, 
        v_head_params=None,
        pi_beta_model=base_model, 
        base_model=base_model, 
        q_head_model=q_head, 
        v_head_model=None, 
        tokenizer=tokenizer, 
        beta=policy_beta, 
        dp_shard_logits=True, 
    )

    policy_prng = jax.random.PRNGKey(0)
    
    def evaluator(inference: GPT2InferenceMask):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        
        all_results = dict()
        interactions = dict()
        
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
        
        with open(test_positions_path, "r") as f:
            test_positions = list(f)
        # test_positions = test_positions[:500]
        test_positions = [position.replace("\n", "").replace("\"", "") for position in test_positions if position != ""]
        
        interactions, results = text_env_eval_chess_positions(
            positions=test_positions,
            policy=policy,
            n_rollouts=policy_n_rollouts, # do multiple, also do no sampling policy 
            verbose=True,
            bsize=policy_bsize,
        )

        if outputs_path is not None:
            create_path(outputs_path)
            with open(os.path.join(outputs_path, 'interactions.pkl'), 'wb') as f:
                pkl.dump(interactions, f)
            with open(os.path.join(outputs_path, 'results.json'), 'w') as f:
                json.dump(jax.tree_util.tree_map(lambda x: float(x), results), f)

        return all_results
        
    
    print(evaluator(
        inference=inference,
    ))

if __name__ == "__main__":
    tyro.cli(main)
