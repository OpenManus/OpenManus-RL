import jax.numpy as jnp
import optax
import jax
from flax.training.train_state import TrainState
from JaxSeq.utils import match_partition_rules
from JaxSeq.checkpointing import load_pytree
from functools import partial
from typing import Union, Callable, Optional
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jaxtyping import PyTree
from JaxSeq.utils import float_to_dtype
import flax.linen as nn
from JaxSeq.utils import multihost_device_put, multihost_device_get
from jax.experimental.pjit import pjit

def get_sharding_from_model(
    model: nn.Module, 
    tree: PyTree, 
) -> Optional[PyTree]:
    if model.config.mesh is not None:
        spec = match_partition_rules(model.config.get_partition_rules(), tree)
        sharding = jax.tree_util.tree_map(lambda ps: NamedSharding(model.config.mesh, ps), spec)
        return sharding
    return None

def shard_params_from_params(
    model: nn.Module, 
    params: PyTree, 
) -> PyTree:
    # get shard spec
    sharding = get_sharding_from_model(model, params)
    assert sharding is not None

    # get sharded params
    params = jax.tree_util.tree_map(lambda x, s: multihost_device_put(x, s), params, sharding)

    return params

def shard_train_state_from_params(
    model: nn.Module, 
    params: PyTree, 
    optim: optax.GradientTransformation, 
) -> TrainState:
    # setup train_state init function
    init_fn = lambda params: partial(TrainState.create, tx=optim, apply_fn=None)(params=params)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, params=params)
    out_shardings = get_sharding_from_model(model, train_state_shape)
    assert out_shardings is not None

    # get sharded train_state
    train_state = pjit(
        init_fn, 
        in_shardings=(out_shardings.params,), 
        out_shardings=out_shardings, 
        donate_argnums=(0,), 
    )(params)

    return train_state

def shard_params_from_config(
    model: nn.Module, 
    prng_key: jax.random.PRNGKeyArray, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
) -> PyTree:
    # setup init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> PyTree:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return params

    # get shard spec
    params_shape = jax.eval_shape(init_fn, prng_key)
    out_shardings = get_sharding_from_model(model, params_shape)
    assert out_shardings is not None

    # get sharded params
    params = jax.jit(
        init_fn, 
        out_shardings=out_shardings, 
    )(prng_key)

    return params

def shard_train_state_from_config(
    model: nn.Module, 
    optim: optax.GradientTransformation, 
    prng_key: jax.random.PRNGKeyArray, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
) -> TrainState:
    
    # setup train_state init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> TrainState:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return TrainState.create(params=params, tx=optim, apply_fn=None)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, prng_key)
    out_shardings = get_sharding_from_model(model, train_state_shape)
    assert out_shardings is not None

    # get sharded train_state
    train_state = jax.jit(
        init_fn, 
        out_shardings=out_shardings, 
    )(prng_key)

    return train_state

def shard_params_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    params_dtype: Union[str, jnp.dtype]=jnp.float32, 
    stream_sharding: bool=True, # shard tensors as they are loaded
) -> PyTree:
    # setup init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> PyTree:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        params = float_to_dtype(params, dtype=params_dtype)
        return params

    # get shard spec
    params_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    sharding = get_sharding_from_model(model, params_shape)
    assert sharding is not None

    # load params with sharding
    with jax.default_device(jax.devices('cpu')[0]):
        params = load_pytree(
            checkpoint_path, 
            target=params_shape, 
            dtype=params_dtype, 
            sharding=sharding if stream_sharding else None, 
        )

    if not stream_sharding:
        params = jax.tree_util.tree_map(lambda x, s: multihost_device_put(x, s), params, sharding)
    return params

def shard_train_state_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], # gets optim from params
    just_params: bool = False, 
    train_state_dtype: Union[str, jnp.dtype]=jnp.float32, 
    stream_sharding: bool=True, # shard tensors as they are loaded
) -> Union[TrainState, PyTree]:
    # setup train_state init function
    def init_fn(prng_key: jax.random.PRNGKeyArray) -> TrainState:
        params = model.init(
            {'params': prng_key}, 
            jnp.zeros((1, model.config.input_dim), dtype=model.dtype), 
            train=True, 
        )['params']
        optim = optim_getter(params)
        return TrainState.create(params=params, tx=optim, apply_fn=None)

    # get shard spec
    train_state_shape = jax.eval_shape(init_fn, jax.random.PRNGKey(0))
    sharding = get_sharding_from_model(model, train_state_shape)
    assert sharding is not None

    # load train_state
    with jax.default_device(jax.devices('cpu')[0]):
        train_state = load_pytree(
            checkpoint_path, 
            target=train_state_shape, 
            dtype=train_state_dtype, 
            sharding=sharding if stream_sharding else None, 
        )
    
    # get sharded params
    if just_params:
        params = train_state.params
        if not stream_sharding:
            params = jax.tree_util.tree_map(lambda x, s: multihost_device_put(x, s), params, sharding.params)
        return params

    # get sharded train_state
    if not stream_sharding:
        train_state = jax.tree_util.tree_map(lambda x, s: multihost_device_put(x, s), train_state, sharding)
    return train_state

def shard_train_state_from_train_state(
    model: nn.Module,     
    train_state: TrainState, 
) -> TrainState:
    # get shard spec
    sharding = get_sharding_from_model(model, train_state)
    assert sharding is not None

    # get sharded train_state
    train_state = jax.tree_util.tree_map(lambda x, s: multihost_device_put(x, s), train_state, sharding)

    return train_state

def copy_sharded_pytree(
    model: nn.Module, 
    pytree: PyTree, 
):
    # define copy func
    def copy_func(x, sharding):
        with jax.default_device(jax.devices('cpu')[0]):
            x = multihost_device_get(x, sharding).copy()
        return multihost_device_put(x, sharding)

    # get shard spec
    sharding = get_sharding_from_model(model, pytree)
    assert sharding is not None

    # copy sharded pytree
    pytree = jax.tree_util.tree_map(
        copy_func, 
        pytree, 
        sharding, 
    )

    return pytree
