from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from flax import struct
from typing import List, Optional, Union, Tuple, Callable, NamedTuple, Dict, Any
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import BlockingStrategy, block_sequences, Padding, Truncation, multihost_device_get
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
import flax.linen as nn
from LLM_RL.utils import get_tensor_stats, unpad_array
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from LLM_RL.environment import TokenTrajectoryChain
from tqdm.auto import tqdm
from LLM_RL.algorithms.ppo.data import PPOData
from LLM_RL.environment import BatchedTextPolicy
from jax.experimental.pjit import pjit

# x
# input = x[1:]
# output = x[:-1]
# if ends on action x[-1] is action, then next window should start with that action, x[0] or next state
# if ends with state, then next window should start with a state, x[0]

# TODO:

# test on some more toy data for multistep / multichain settings
# add in ILQL / others
# clean code

# KL Controllers
# adapted from: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py

class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁

class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


def ppo_loss_fn(
    attention_mask: jax.Array, # [batch, time-1] – output is masked; shift x[1:]
    logprobs: jax.Array, # [batch, time-1] – logprob of output produced; shift x[1:]
    values: jax.Array, # [batch, time-1] – value of current state; shift x[:-1]
    should_take_action: jax.Array, # [batch, time-1] – is output produced by action; shift x[1:]
    old_logprobs: jax.Array, # [batch, time-1] – logprob of output produced; shift x[1:]
    old_values: jax.Array, # [batch, time-1] – value of current state; shift x[:-1]
    old_advantages: jax.Array, # [batch, time-1] – advantage of output produced; shift x[1:]
    old_returns: jax.Array, # [batch, time-1] – return of current state; shift x[:-1]
    *, 
    cliprange_value: Union[float, jax.Array], 
    cliprange: Union[float, jax.Array], 
    value_loss_coef: Union[float, jax.Array], 
) -> Tuple[jax.Array, Dict[str, Any]]:
    """PPO objective function.
    References:
    - https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
    - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
    """
    mask = should_take_action.astype(jnp.float32) * attention_mask
    n = mask.sum()
    
    values_clipped = jnp.clip(
        values, 
        old_values - cliprange_value, 
        old_values + cliprange_value, 
    )

    vf_loss1 = (values - old_returns) ** 2
    vf_loss2 = (values_clipped - old_returns) ** 2
    vf_loss = 0.5 * jnp.sum(jnp.maximum(vf_loss1, vf_loss2) * mask) / n
    vf_clipfrac = jnp.sum((vf_loss2 > vf_loss1).astype(jnp.float32) * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = jnp.exp(log_ratio)
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    approx_kl = jnp.sum((ratio - 1) - log_ratio) / n

    pg_loss1 = -old_advantages * ratio
    pg_loss2 = -old_advantages * jnp.clip(
        ratio, 
        1.0 - cliprange, 
        1.0 + cliprange, 
    )
    pg_loss = jnp.sum(jnp.maximum(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = jnp.sum((pg_loss2 > pg_loss1).astype(jnp.float32) * mask) / n

    loss = pg_loss + value_loss_coef * vf_loss

    logs = dict(
        losses=dict(
            total_loss=loss, 
            policy_loss=pg_loss, 
            value_loss=vf_loss, 
        ), 
        values=dict(
            get_tensor_stats(values, mask, n), 
            values_error=jnp.sum(((values - old_returns) * mask) ** 2) / n, 
            clipfrac=vf_clipfrac, 
        ), 
        old_values=get_tensor_stats(old_values, mask, n), 
        returns=get_tensor_stats(old_returns, mask, n), 
        policy=dict(
            approx_kl=approx_kl, 
            clipfrac=pg_clipfrac, 
        ), 
        ratio=(ratio * mask).sum() / n, 
        padding_percentage=n / mask.size, 
    )

    return loss, logs

class PPOTrain(struct.PyTreeNode):
    policy_train_state: TrainState
    value_head_train_state: TrainState
    policy_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    value_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     policy_train_state: TrainState, 
    #     value_head_train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     old_logprobs: jax.Array, 
    #     old_values: jax.Array, 
    #     old_advantages: jax.Array, 
    #     old_returns: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     bc_data_input_ids: Optional[jax.Array], 
    #     bc_data_input_attention_mask: Optional[jax.Array], 
    #     bc_data_input_position_ids: Optional[jax.Array], 
    #     bc_data_input_training_mask: Optional[jax.Array], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, TrainState, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        old_logprobs: jax.Array, # [batch, time-1]
        old_values: jax.Array, # [batch, time-1]
        old_advantages: jax.Array, # [batch, time-1]
        old_returns: jax.Array, # [batch, time-1]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        bc_data_input_ids: Optional[jax.Array]=None, 
        bc_data_input_attention_mask: Optional[jax.Array]=None, 
        bc_data_input_position_ids: Optional[jax.Array]=None, 
        bc_data_input_training_mask: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[PPOTrain, jax.Array, PyTree]:
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        if bc_data_input_ids is not None:
            bc_data_input_attention_mask, bc_data_input_position_ids = initialize_attn_mask_pos_ids(
                bc_data_input_ids, 
                self.tokenizer.pad_token_id, 
                bc_data_input_attention_mask, 
                bc_data_input_position_ids, 
            )
            assert bc_data_input_training_mask is not None
        
        policy_train_state, value_head_train_state, loss, logs = self._step(
            self.policy_train_state, 
            self.value_head_train_state, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            old_logprobs, 
            old_values, 
            old_advantages, 
            old_returns, 
            prng_key, 
            bc_data_input_ids, 
            bc_data_input_attention_mask, 
            bc_data_input_position_ids, 
            bc_data_input_training_mask, 
            train, 
        )

        return self.replace(
            policy_train_state=policy_train_state, 
            value_head_train_state=value_head_train_state, 
        ), loss, logs

def get_action_state_next_state_idxs(
    should_take_action: np.ndarray, # [t-1]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_idxs = np.where(should_take_action)[0]
    state_idxs = np.where(should_take_action)[0]

    is_next_state = should_take_action.copy()
    is_next_state[np.argmax(is_next_state.astype(np.int32), axis=0)] = False
    is_next_state = np.concatenate((is_next_state, (should_take_action.sum(axis=0) > 0)[None]), axis=0)
    next_state_idxs = np.where(is_next_state)[0]

    assert action_idxs.shape == state_idxs.shape == next_state_idxs.shape

    return action_idxs, state_idxs, next_state_idxs

def whiten(xs: jax.Array, shift_mean=True) -> jax.Array:
    """Whitens values"""
    mean, var = jnp.mean(xs), jnp.var(xs)
    whitened = (xs - mean) * jnp.reciprocal(jnp.sqrt(var+1e-8))
    if not shift_mean:
        whitened += mean
    return whitened

def get_advantages_and_returns(
    state_values: np.ndarray, # [b, t-1]
    next_state_values: np.ndarray, # [b, t-1]
    action_rewards: np.ndarray, # [b, t-1]
    *, 
    gamma: Union[float, np.ndarray], 
    lam: Union[float, np.ndarray], 
    use_whitening: bool = True, 
) -> Tuple[np.ndarray, np.ndarray]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.
    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
    Args:
        values: Tensor of shape (batch_size, response_size)
        rewards: Tensor of shape (batch_size, response_size)
        response_length: Length of the response sequence
        use_whitening: Whether to use whitening (ie. normalize advantages) or not
    
    References:
    - https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
    """
    assert state_values.shape == next_state_values.shape == action_rewards.shape
    n = state_values.shape[1]
    
    lastgaelam = 0
    advantages_reversed = []
    for t in reversed(range(n)):
        delta = action_rewards[:, t] + gamma * next_state_values[:, t] - state_values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + state_values
    if use_whitening:
        advantages = whiten(advantages)
    return advantages, returns

class CombinedTokenTrajectoryChain(NamedTuple):
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    rewards: np.ndarray
    should_take_action: np.ndarray
    done: Union[bool, np.ndarray]
    chunk_lens: List[int]

    @classmethod
    def from_token_trajectory_chain(
        cls, 
        token_trajectory_chain: TokenTrajectoryChain, 
        max_length: Optional[int]=None, 
    ) -> CombinedTokenTrajectoryChain:
        token_trajectories = token_trajectory_chain.to_list()
        assert len(token_trajectories) > 0, "token_trajectory_chain must have at least one token_trajectory"

        if max_length is None:
            max_length = max([tt.tokens.shape[0] for tt in token_trajectories])+1
        
        # double check dones
        assert not any([tt.done for tt in token_trajectories[:-1]]), "done can only be true at the end of the chain"
        
        # check truncation conditions
        for i in range(len(token_trajectories)):
            # check that the trajectory is not truncated or that it doesn't end with a state and start with an action
            # we can't calculate the advantage if the trajectory is truncated and there are later actions
            no_trunc = (token_trajectories[i].tokens.shape[0]-1) <= max_length
            ends_with_state = (not np.any(token_trajectories[i].is_action[1:][max_length:]))
            next_starts_with_action = i < len(token_trajectories)-1 and token_trajectories[i+1].is_action[0]

            assert not (ends_with_state and next_starts_with_action), 'trajectory truncation error'
            assert no_trunc or ends_with_state, 'trajectory truncation error'

        return cls(
            input_tokens=np.concatenate([tt.tokens[:-1][:max_length] for tt in token_trajectories], axis=0), 
            output_tokens=np.concatenate([tt.tokens[1:][:max_length] for tt in token_trajectories], axis=0), 
            rewards=np.concatenate([tt.reward[1:][:max_length] for tt in token_trajectories], axis=0), 
            should_take_action=np.concatenate([tt.is_action[1:][:max_length] for tt in token_trajectories], axis=0), 
            done = token_trajectories[-1].done, 
            chunk_lens=[min(tt.tokens.shape[0]-1, max_length) for tt in token_trajectories], 
        )
    
    def unroll_arr(
        self, 
        arr: np.ndarray, 
    ) -> List[np.ndarray]:
        assert arr.shape[0] == self.input_tokens.shape[0]
        return np.split(arr, np.cumsum(self.chunk_lens)[:-1], axis=0)

class PPOForwardOutput(NamedTuple):
    initial_policy_raw_output: FlaxCausalLMOutput
    policy_raw_output: FlaxCausalLMOutput
    values: jax.Array

class PPOInference(struct.PyTreeNode):
    initial_policy_params: Optional[PyTree]
    policy_params: PyTree
    value_head_params: PyTree
    initial_policy_model: Optional[FlaxPreTrainedModel] = struct.field(pytree_node=False) # corresponds to initial_policy_params
    policy_model: FlaxPreTrainedModel = struct.field(pytree_node=False) # corresponds to policy_params
    value_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _eval_loss: Optional[Callable] = struct.field(pytree_node=False, default=None)
    
    # def _forward(
    #     initial_policy_params: PyTree, 
    #     policy_params: PyTree, 
    #     value_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     initial_policy_output_attentions: Optional[bool]=None, 
    #     initial_policy_output_hidden_states: Optional[bool]=None, 
    #     policy_output_attentions: Optional[bool]=None, # no policy_output_hidden_states option because this is required
    #     train: bool=False, 
    # ) -> PPOForwardOutput:
    #     raise NotImplementedError

    # def _eval_loss(
    #     policy_params: PyTree, 
    #     value_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     old_logprobs: jax.Array, 
    #     old_values: jax.Array, 
    #     old_advantages: jax.Array, 
    #     old_returns: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     bc_data_input_ids: Optional[jax.Array], 
    #     bc_data_input_attention_mask: Optional[jax.Array], 
    #     bc_data_input_position_ids: Optional[jax.Array], 
    #     bc_data_input_training_mask: Optional[jax.Array], 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError

    @staticmethod
    @pjit
    def token_logprobs_from_logits(
        logits: jax.Array, 
        input_ids: jax.Array, 
    ) -> jax.Array:
        token_log_probs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1].astype(jnp.float32), input_ids[:, 1:])
        return token_log_probs
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        initial_policy_output_attentions: Optional[bool]=None, 
        initial_policy_output_hidden_states: Optional[bool]=None, 
        policy_output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> PPOForwardOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.initial_policy_params, 
            self.policy_params, 
            self.value_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            initial_policy_output_attentions, 
            initial_policy_output_hidden_states, 
            policy_output_attentions, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        initial_policy_output_attentions: Optional[bool]=None, 
        initial_policy_output_hidden_states: Optional[bool]=None, 
        policy_output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> PPOForwardOutput:
        if token_process is None:
            token_process = lambda x: x
        # tokenize
        tokens = [token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            initial_policy_output_attentions=initial_policy_output_attentions, 
            initial_policy_output_hidden_states=initial_policy_output_hidden_states, 
            policy_output_attentions=policy_output_attentions, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs
    
    def get_ppo_data_from_token_trajectory_chain(
        self, 
        token_trajectory_chains: List[TokenTrajectoryChain], 
        bsize: int, 
        max_length: Optional[int]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        verbose: bool=True, 
        *, 
        gamma: Union[float, jax.Array], 
        lam: Union[float, jax.Array], 
        kl_weight: Union[float, jax.Array], 
        use_advantage_whitening: bool=True,
        use_new_advantage_whitening: bool=False,
    ) -> Tuple[List[PPOData], np.ndarray]:
        assert self.initial_policy_model is not None and self.initial_policy_params is not None
        n_chains = len(token_trajectory_chains)

        tokens = []
        combined_token_trajectory_chains = []
        for token_trajectory_chain in token_trajectory_chains:
            # max_length - 1 because we clip endpoints.
            # not sure if it's more optimal to do -1 here or +1 in the forward function.
            combined_token_trajectory_chains.append(
                CombinedTokenTrajectoryChain.from_token_trajectory_chain(
                    token_trajectory_chain, 
                    max_length=max_length-1 if max_length is not None else None, 
                )
            )
            tokens.extend(list(map(lambda x: x.tokens, token_trajectory_chain.to_list())))

        tokens = block_sequences(
            tokens, 
            pad_value=self.tokenizer.pad_token_id, 
            dtype=np.int32, 
            blocking_strategy=BlockingStrategy(
                padding=Padding.RIGHT, 
                truncation=Truncation.RIGHT, 
                max_length=max_length, 
            ), 
        )

        # get values, logits from forward pass
        initial_policy_logprobs, policy_logprobs, values = [], [], []
        print("getting log probs...")
        for i in tqdm(range(0, len(tokens), bsize), disable=not verbose):
            tokens_batch = jnp.asarray(tokens[i:(i+bsize)], dtype=jnp.int32)
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            forward_batch_output = self.forward(
                tokens_batch, 
                train=train, 
                prng_key=new_key, 
            )

            initial_policy_logits = forward_batch_output.initial_policy_raw_output.logits
            initial_policy_logprob = self.token_logprobs_from_logits(initial_policy_logits, tokens_batch)
            initial_policy_logprob = np.asarray(multihost_device_get(
                initial_policy_logprob, 
                mesh=self.initial_policy_model.config.mesh, 
            ))

            policy_logits = forward_batch_output.policy_raw_output.logits
            policy_logprob = self.token_logprobs_from_logits(policy_logits, tokens_batch)
            policy_logprob = np.asarray(multihost_device_get(
                policy_logprob, 
                mesh=self.policy_model.config.mesh, 
            ))

            initial_policy_logprobs.append(initial_policy_logprob)
            policy_logprobs.append(policy_logprob)
            values.append(np.asarray(forward_batch_output.values))
        
        initial_policy_logprobs = np.concatenate(initial_policy_logprobs, axis=0)
        policy_logprobs = np.concatenate(policy_logprobs, axis=0)
        values = np.concatenate(values, axis=0)

        batch_sections = list(map(lambda x: len(x.chunk_lens), combined_token_trajectory_chains))
        mask_split_by_chain = np.split((tokens != self.tokenizer.pad_token_id), np.cumsum(batch_sections)[:-1], axis=0)

        initial_policy_logprobs_split_by_chain = np.split(initial_policy_logprobs, np.cumsum(batch_sections)[:-1], axis=0)
        policy_logprobs_split_by_chain = np.split(policy_logprobs, np.cumsum(batch_sections)[:-1], axis=0)
        values_split_by_chain = np.split(values, np.cumsum(batch_sections)[:-1], axis=0)
        
        initial_policy_logprobs_chains = [
            np.concatenate(list(map(lambda x, m: unpad_array(x, m), item, mask[:, 1:])), axis=0)
            for mask, item in zip(mask_split_by_chain, initial_policy_logprobs_split_by_chain)
        ]
        policy_logprobs_chains = [
            np.concatenate(list(map(lambda x, m: unpad_array(x, m), item, mask[:, 1:])), axis=0)
            for mask, item in zip(mask_split_by_chain, policy_logprobs_split_by_chain)
        ]

        values_chains = [
            np.concatenate(list(map(lambda x, m: unpad_array(x, m)[:-1], item, mask)), axis=0)
            for mask, item in zip(mask_split_by_chain, values_split_by_chain)
        ]
        # add last value for final step bootstrapping
        last_values_chains = [
            unpad_array(item[-1], mask[-1])[-1]
            for mask, item in zip(mask_split_by_chain, values_split_by_chain)
        ]
        values_chains = [
            np.concatenate((item, last_values_chains[i][None]*(1.0-float(combined_token_trajectory_chains[i].done))), axis=0)
            for i, item in enumerate(values_chains)
        ]

        log_ratio = [
            (policy_logprob - initial_policy_logprob) * chain.should_take_action.astype(np.float32)
            for initial_policy_logprob, policy_logprob, chain in zip(initial_policy_logprobs_chains, policy_logprobs_chains, combined_token_trajectory_chains)
        ]

        valid_log_ratio_idxs = np.argwhere(np.concatenate(list(map(lambda chain: chain.should_take_action.astype(np.float32).reshape(-1), combined_token_trajectory_chains)), axis=0))[:, 0]
        all_log_ratio = np.concatenate(list(map(lambda x: x.reshape(-1), log_ratio)), axis=0)[valid_log_ratio_idxs]
        all_kls = np.exp(all_log_ratio) - 1 - all_log_ratio
        # add kl penalty to reward
        for i in range(n_chains):
            combined_token_trajectory_chains[i] = combined_token_trajectory_chains[i]._replace(
                rewards=combined_token_trajectory_chains[i].rewards - kl_weight * log_ratio[i], 
            )

        all_advantages, all_returns = [], []
        for i in range(n_chains):
            action_idxs, state_idxs, next_state_idxs = get_action_state_next_state_idxs(
                combined_token_trajectory_chains[i].should_take_action, 
            )

            state_values = values_chains[i][state_idxs]
            next_state_values = values_chains[i][next_state_idxs]
            action_rewards = combined_token_trajectory_chains[i].rewards[action_idxs]

            advantages, returns = get_advantages_and_returns(
                state_values=state_values[None], 
                next_state_values=next_state_values[None], 
                action_rewards=action_rewards[None], 
                gamma=gamma, 
                lam=lam, 
                use_whitening=False, 
            )

            all_advantages.append(advantages[0])
            all_returns.append(returns[0])
        
        # do advantage whitening over the full batch
        if use_advantage_whitening:
            whitened_advantages = whiten(np.concatenate(all_advantages, axis=0), shift_mean=True)
            curr_pos = 0
            for i in range(n_chains):
                curr_len = all_advantages[i].shape[0]
                all_advantages[i] = whitened_advantages[curr_pos:(curr_pos+curr_len)]
                curr_pos += curr_len

        advantage_chains, return_chains = [], []
        for i in range(n_chains):
            action_idxs, state_idxs, next_state_idxs = get_action_state_next_state_idxs(
                combined_token_trajectory_chains[i].should_take_action, 
            )

            all_advantages.append(advantages[0])
            all_returns.append(returns[0])
        
        # do advantage whitening over the full batch
        if use_new_advantage_whitening:
            whitened_advantages = whiten(np.concatenate(all_advantages, axis=0), shift_mean=True)
            curr_pos = 0
            for i in range(n_chains):
                curr_len = all_advantages[i].shape[0]
                all_advantages[i] = whitened_advantages[curr_pos:(curr_pos+curr_len)]
                curr_pos += curr_len

        advantage_chains, return_chains = [], []
        for i in range(n_chains):
            action_idxs, state_idxs, next_state_idxs = get_action_state_next_state_idxs(
                combined_token_trajectory_chains[i].should_take_action, 
            )

            advantage_chain = np.zeros((values_chains[i].shape[0]-1,), dtype=np.float32)
            advantage_chain[action_idxs] = all_advantages[i]

            return_chain = np.zeros((values_chains[i].shape[0]-1,), dtype=np.float32)
            return_chain[action_idxs] = all_returns[i]

            advantage_chains.append(advantage_chain)
            return_chains.append(return_chain)
        
        ppo_datas = []
        for i in range(n_chains):
            input_ids_chunks = list(map(lambda x: x.tokens[:max_length], token_trajectory_chains[i].to_list())) # trunc to max_length
            should_take_action_chunks = combined_token_trajectory_chains[i].unroll_arr(combined_token_trajectory_chains[i].should_take_action)
            old_logprobs_chunks = combined_token_trajectory_chains[i].unroll_arr(policy_logprobs_chains[i])
            old_values = combined_token_trajectory_chains[i].unroll_arr(values_chains[i][:-1])
            old_advantages = combined_token_trajectory_chains[i].unroll_arr(advantage_chains[i])
            old_returns = combined_token_trajectory_chains[i].unroll_arr(return_chains[i])

            for chunk_idx in range(len(combined_token_trajectory_chains[i].chunk_lens)):
                ppo_datas.append(PPOData(
                    input_ids=input_ids_chunks[chunk_idx], 
                    should_take_action=should_take_action_chunks[chunk_idx], 
                    old_logprobs=old_logprobs_chunks[chunk_idx], 
                    old_values=old_values[chunk_idx], 
                    old_advantages=old_advantages[chunk_idx], 
                    old_returns=old_returns[chunk_idx], 
                ))

        return ppo_datas, all_kls

    def get_ppo_data_from_text_trajectory_chain(
        self, 
        text_trajectory_chains: List[TokenTrajectoryChain], 
        bsize: int, 
        max_length: Optional[int]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        verbose: bool=True, 
        *, 
        gamma: Union[float, jax.Array], 
        lam: Union[float, jax.Array], 
        kl_weight: Union[float, jax.Array], 
        use_advantage_whitening: bool=True,
        use_new_advantage_whitening: bool=False,
    ) -> Tuple[List[PPOData], np.ndarray]:
        
        token_trajectory_chains = [
            TokenTrajectoryChain.from_text_trajectory_chain(
                item, 
                self.tokenizer, 
                token_process=token_process, 
            ) for item in text_trajectory_chains
        ]

        return self.get_ppo_data_from_token_trajectory_chain(
            token_trajectory_chains=token_trajectory_chains, 
            bsize=bsize, 
            max_length=max_length, 
            train=train, 
            prng_key=prng_key, 
            verbose=verbose, 
            gamma=gamma, 
            lam=lam, 
            kl_weight=kl_weight, 
            use_advantage_whitening=use_advantage_whitening, 
            use_new_advantage_whitening=use_new_advantage_whitening,
        )
    
    def get_ppo_data_from_text_trajectory_chain_iterble(
        self, 
        text_trajectory_chains: List[TokenTrajectoryChain], 
        bsize: int, 
        max_length: Optional[int]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        verbose: bool=True, 
        *, 
        gamma: Union[float, jax.Array], 
        lam: Union[float, jax.Array], 
        kl_weight: Union[float, jax.Array], 
        use_advantage_whitening: bool=True, 
    ) -> Tuple[List[PPOData], np.ndarray]:
        
        token_trajectory_chains = [
            TokenTrajectoryChain.from_text_trajectory_chain(
                item, 
                self.tokenizer, 
                token_process=token_process, 
            ) for item in text_trajectory_chains
        ]

        return self.get_ppo_data_from_token_trajectory_chain(
            token_trajectory_chains=token_trajectory_chains, 
            bsize=bsize, 
            max_length=max_length, 
            train=train, 
            prng_key=prng_key, 
            verbose=verbose, 
            gamma=gamma, 
            lam=lam, 
            kl_weight=kl_weight, 
            use_advantage_whitening=use_advantage_whitening, 
        )
    
    def eval_loss(
        self, 
        input_ids: jax.Array, 
        should_take_action: jax.Array, 
        old_logprobs: jax.Array, 
        old_values: jax.Array, 
        old_advantages: jax.Array, 
        old_returns: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        bc_data_input_ids: Optional[jax.Array]=None, 
        bc_data_input_attention_mask: Optional[jax.Array]=None, 
        bc_data_input_position_ids: Optional[jax.Array]=None, 
        bc_data_input_training_mask: Optional[jax.Array]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        if bc_data_input_ids is not None:
            bc_data_input_attention_mask, bc_data_input_position_ids = initialize_attn_mask_pos_ids(
                bc_data_input_ids, 
                self.tokenizer.pad_token_id, 
                bc_data_input_attention_mask, 
                bc_data_input_position_ids, 
            )
            assert bc_data_input_training_mask is not None

        return self._eval_loss(
            self.policy_params, 
            self.value_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            old_logprobs, 
            old_values, 
            old_advantages, 
            old_returns, 
            prng_key, 
            bc_data_input_ids, 
            bc_data_input_attention_mask, 
            bc_data_input_position_ids, 
            bc_data_input_training_mask, 
            train, 
        )

    # def eval_loss_from_ppo_data(
    #     self, 
    #     ppo_data: List[PPOData], 
    #     blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
        
    #     ppo_data_batch_dict = PPOData.block(
    #         ppo_data, 
    #         blocking_strategy, 
    #         self.tokenizer, 
    #     )

    #     return self.eval_loss(
    #         **ppo_data_batch_dict, 
    #         prng_key=prng_key, 
    #         train=train, 
    #     )

class PPOPolicy(BatchedTextPolicy):
    def set_params(self, policy_params: PyTree) -> None:
        raise NotImplementedError
