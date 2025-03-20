from __future__ import annotations
import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Optional, Union, Callable, Tuple, Dict, Any
import re
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
from jaxtyping import PyTree
from enum import Enum
import optax
from flax.training.train_state import TrainState
from LLM_RL.heads.shard_heads import shard_params_from_config, shard_train_state_from_params, shard_train_state_from_checkpoint, shard_params_from_checkpoint, get_sharding_from_model
from flax.core import freeze, unfreeze
from JaxSeq.bucket_manager import open_with_bucket as open
import json
import os
from LLM_RL.heads.base import HeadConfig
from JaxSeq.utils import multihost_device_get, multihost_device_put

class ModelLoadMode(Enum):
    CONFIG = 'config'
    TRAIN_STATE = 'train_state'
    TRAIN_STATE_PARAMS = 'train_state_params'
    PARAMS = 'params'

    @staticmethod
    def match_load_mode(load_mode: Union[ModelLoadMode, str], target: ModelLoadMode):
        if isinstance(load_mode, str):
            return load_mode == target.value
        return load_mode == target

def pad_outputs(
    params: PyTree, 
    model: nn.Module, 
    pad_to_output_dim: int, 
    dtype: jnp.dtype=jnp.float32, 
) -> PyTree:
    old_size = model.config.output_dim
    model.config.output_dim = pad_to_output_dim
    print(f'Padding outputs from size {old_size} to size {model.config.output_dim}.')
    # pad outputs
    sharding = get_sharding_from_model(model, params)
    return model.pad_outputs(params, param_sharding=sharding, dtype=dtype)

# basic linear head

class LinearHeadConfig(HeadConfig):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        use_bias: bool=True, 
        unpadded_output_dim: Optional[int]=None, 
        initializer_range: Optional[int]=None, 
        bias_init: Optional[float]=None, 
        mesh: Optional[jax.sharding.Mesh]=None, 
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.bias_init = bias_init
        self.mesh = mesh
        self.unpadded_output_dim = unpadded_output_dim
        if self.unpadded_output_dim is None:
            self.unpadded_output_dim = self.output_dim
        super().__init__()
    
    @staticmethod
    def get_partition_rules():
        return [
            (re.escape("['dense']['kernel']"), PS()), 
            (re.escape("['dense']['bias']"), PS()), 
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        if self.mesh is None:
            return super().to_dict()
        else:
            new_conf = LinearHeadConfig(**self.__dict__)
            new_conf.mesh = None
            return new_conf.to_dict()

class LinearHead(nn.Module):
    config: LinearHeadConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        if self.config.initializer_range is None:
            kernel_initalizer = jax.nn.initializers.lecun_normal()
        else:
            kernel_initalizer = jax.nn.initializers.normal(self.config.initializer_range)
        
        if self.config.bias_init is None:
            bias_initalizer = jax.nn.initializers.zeros
        else:
            bias_initalizer = jax.nn.initializers.constant(self.config.bias_init)
        
        self.dense = nn.Dense(
            features=self.config.output_dim, 
            use_bias=self.config.use_bias, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            precision=self.precision, 
            kernel_init=kernel_initalizer, 
            bias_init=bias_initalizer, 
        )

    def __call__(
        self, 
        x: jax.Array, 
        *, 
        train: bool, 
    ) -> jax.Array:
        x = self.dense(x)
        return x
    
    def pad_outputs(self, params: PyTree, param_sharding: Optional[PyTree]=None, dtype: jnp.dtype=jnp.float32) -> PyTree:
        assert params["dense"]["kernel"].shape == (self.config.input_dim, self.config.unpadded_output_dim), f"param shape doesn't match expected got {params['dense']['kernel'].shape} expected {(self.config.input_dim, self.config.unpadded_output_dim)}"
        assert params["dense"]["bias"].shape == (self.config.unpadded_output_dim,), f"param shape doesn't match expected got {params['dense']['bias'].shape} expected {(self.config.unpadded_output_dim,)}"
        if param_sharding is not None:
            params["dense"]["kernel"] = multihost_device_get(
                params["dense"]["kernel"], 
                param_sharding["dense"]["kernel"], 
            )
        out_kernel = jnp.zeros((self.config.input_dim, self.config.output_dim), dtype=dtype)
        params["dense"]["kernel"] = out_kernel.at[:, :self.config.unpadded_output_dim].set(params["dense"]["kernel"])
        if param_sharding is not None:
            params["dense"]["kernel"] = multihost_device_put(
                params["dense"]["kernel"], 
                param_sharding["dense"]["kernel"], 
            )
        if self.config.use_bias:
            if param_sharding is not None:
                params["dense"]["bias"] = multihost_device_get(
                    params["dense"]["bias"], 
                    param_sharding["dense"]["bias"], 
                )
            out_bias = jnp.zeros((self.config.output_dim,), dtype=dtype)
            params["dense"]["bias"] = out_bias.at[:self.config.unpadded_output_dim].set(params["dense"]["bias"])
            if param_sharding is not None:
                params["dense"]["bias"] = multihost_device_put(
                    params["dense"]["bias"], 
                    param_sharding["dense"]["bias"], 
                )
        return params

def load_train_state_from_config(
    model_config: LinearHeadConfig, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    mesh: Mesh, # should be shape (dp, mp)
    prng_key: jax.random.PRNGKeyArray, 
    pad_to_output_dim: Optional[int]=None, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, LinearHead]:
    
    model = LinearHead(model_config, dtype=model_dtype)
    model.config.mesh = mesh
    # shard params
    params = freeze(shard_params_from_config(model, prng_key, params_dtype=params_dtype))
    # pad outputs
    if pad_to_output_dim is not None:
        params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
    # shard train_state
    train_state = shard_train_state_from_params(model, params, optim_getter(params))

    return train_state, model

def load_train_state(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    mesh: Mesh, # should be shape (dp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    pad_to_output_dim: Optional[int]=None, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, LinearHead]:
    
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        train_state, model = load_train_state_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            optim_getter=optim_getter, 
            mesh=mesh, 
            prng_key=prng_key, 
            pad_to_output_dim=pad_to_output_dim, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        model = LinearHead(model_config, dtype=model_dtype)
        model.config.mesh = mesh
        # shard and pad embeddings
        if pad_to_output_dim is None:
            # if no padding, just load train_state, shard as well
            train_state = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=False, train_state_dtype=params_dtype)
        else:
            # if padding, load params, pad, shard
            params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
            params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
            train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE_PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        model = LinearHead(model_config, dtype=model_dtype)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
        # pad outputs
        if pad_to_output_dim is not None:
            params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        model = LinearHead(model_config, dtype=model_dtype)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_params_from_checkpoint(model, os.path.join(model_load_path, 'params.msgpack'), params_dtype=params_dtype)
        # pad outputs
        if pad_to_output_dim is not None:
            params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return train_state, model

def load_params_from_config(
    model_config: LinearHeadConfig, 
    model_dtype: Union[str, jnp.dtype], 
    mesh: Mesh, # should be shape (dp, mp)
    prng_key: jax.random.PRNGKeyArray, 
    pad_to_output_dim: Optional[int]=None, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, LinearHead]:
    
    model = LinearHead(model_config, dtype=model_dtype)
    model.config.mesh = mesh
    # shard params
    params = shard_params_from_config(model, prng_key, params_dtype=params_dtype)
    # pad outputs
    if pad_to_output_dim is not None:
        params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
    
    return params, model

def load_params(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    mesh: Mesh, # should be shape (dp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    pad_to_output_dim: Optional[int]=None, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, LinearHead]:
    
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        params, model = load_params_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            mesh=mesh, 
            prng_key=prng_key, 
            pad_to_output_dim=pad_to_output_dim, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = LinearHeadConfig.from_dict(json.load(f))
        model = LinearHead(model_config, dtype=model_dtype)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_params_from_checkpoint(model, os.path.join(model_load_path, 'params.msgpack'), params_dtype=params_dtype)
        # pad outputs
        if pad_to_output_dim is not None:
            params = freeze(pad_outputs(unfreeze(params), model, pad_to_output_dim, dtype=params_dtype))
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return params, model
