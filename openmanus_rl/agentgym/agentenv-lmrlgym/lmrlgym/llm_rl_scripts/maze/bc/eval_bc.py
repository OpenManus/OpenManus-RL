from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import load_mesh, create_path
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation
import os
from JaxSeq.models.gpt2.interface import GPT2InferenceMask
from JaxSeq.models.gpt2.load import ModelLoadMode, load_params
import pickle as pkl
import json
from transformers.generation import GenerationConfig
import re
from LLM_RL.algorithms.ppo.gpt2.interface import GPT2PPOPolicy
from LLM_RL.environment import text_env_eval
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env, maze_solver
from collections import defaultdict
import numpy as np
from LLM_RL.algorithms.ppo.reranker_policy import ReRankerSamplePolicy, ReRankerPolicy
from LLM_RL.algorithms.ppo.score_fn import build_bc_score_fn
from llm_rl_scripts.maze.env.env import maze_proposal_function
from flax.traverse_util import flatten_dict, unflatten_dict
from LLM_RL.environment import Text
from llm_rl_scripts.maze.env.env import describe_observation_give_position

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str,

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

    maze_name:str="double_t_maze",
    fully_observed:bool=True, 
    maze_last_k: int=1,
    maze_reward_function: str="standard_reward",

    do_accuracy_eval: bool=True,
    do_reward_eval: bool=True,
    use_reranker_for_reward_eval: bool=False,

    force_pad_embeddings: bool=False,
):
    input_args = locals()
    print(input_args)
    
    if fully_observed: 
        describe_function = "describe_observation_give_position"
    else:
        describe_function = "describe_observation_only_walls"

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})



    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    env = setup_maze_env(
        maze_name=maze_name,
        describe_function=describe_function,
        reward_function=maze_reward_function,
        last_k=maze_last_k,
    )
    possible_positions = list(zip(*np.where(env.maze==0)))
    for goal in env.valid_goals:
        possible_positions.remove(tuple(goal.tolist()))
    optimal_policy = maze_solver(1-env.maze, list(map(tuple, env.valid_goals.tolist())))

    model_prng_key = jax.random.PRNGKey(2)
    params, model = load_params(
        model_load_mode=model_load_mode, 
        model_load_path=model_load_path if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )

    inference = GPT2InferenceMask.load_inference(
        params=params, 
        model=model, 
        tokenizer=tokenizer, 
    )

    policy_prng = jax.random.PRNGKey(0)
    def evaluator(inference: GPT2InferenceMask):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        
        all_results = dict()
        interactions = dict()

        if do_reward_eval:
            if use_reranker_for_reward_eval:
                if policy_do_sample:
                    policy = ReRankerSamplePolicy(
                        proposal_fn=maze_proposal_function, 
                        score_fn=build_bc_score_fn(
                            inference=inference,
                            tokenizer=tokenizer,
                            max_length=policy_max_input_length+policy_max_output_length,
                            bsize=policy_bsize,
                        ),
                    )
                else:
                    policy = ReRankerPolicy(
                        proposal_fn=maze_proposal_function,
                        score_fn=build_bc_score_fn(
                            inference=inference,
                            tokenizer=tokenizer,
                            max_length=policy_max_input_length+policy_max_output_length,
                            bsize=policy_bsize,
                        ),
                    )
            else:
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

            results = dict()
            avg_dict = defaultdict(float)
            for position in possible_positions:
                position = tuple(position)
                interactions[str(position)], results[str(position)] = text_env_eval(
                    env=env,
                    policy=policy,
                    n_rollouts=policy_n_rollouts, # do multiple, also do no sampling policy 
                    verbose=True,
                    env_options={"init_position": position},
                    bsize=policy_bsize,
                )
                for k, v in flatten_dict(results[str(position)]).items():
                    avg_dict[k] += v
            for k, v in avg_dict.items():
                avg_dict[k] = v/len(possible_positions)
            results["avg_reward"] = unflatten_dict(dict(avg_dict))

            all_results["reward_eval"] = results
        
        if do_accuracy_eval:
            results = dict()
            policy = ReRankerPolicy(
                proposal_fn=maze_proposal_function,
                score_fn=build_bc_score_fn(
                    inference=inference,
                    tokenizer=tokenizer,
                    max_length=policy_max_input_length+policy_max_output_length,
                    bsize=policy_bsize,
                ),
            )

            num_correct = 0
            for position in possible_positions:
                print(position, num_correct)
                env.position = position
                observation = describe_observation_give_position(env.maze, position, env.goal)
                text_history = (Text(observation, False),)
                output = policy.act(text_history)
                prediction = output[-1].text

                if prediction.lower().strip() == optimal_policy[tuple(position)].lower().strip():
                    num_correct += 1
                    results[str(position)] = True
                else:
                    results[str(position)] = False
            
            move_accuracy = num_correct/len(possible_positions)
            results["avg_accuracy"] = move_accuracy

            all_results["move_accuracy"] = results
        
        if outputs_path is not None:
            create_path(outputs_path)
            if do_reward_eval:
                with open(os.path.join(outputs_path, 'interactions.pkl'), 'wb') as f:
                    pkl.dump(interactions, f)
            with open(os.path.join(outputs_path, 'results.json'), 'w') as f:
                json.dump(jax.tree_util.tree_map(lambda x: float(x), all_results), f)

        return all_results
    
    print(evaluator(
        inference=inference,
    ))

if __name__ == "__main__":
    tyro.cli(main)
