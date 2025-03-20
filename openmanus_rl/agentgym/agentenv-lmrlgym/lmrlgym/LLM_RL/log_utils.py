from collections import namedtuple
from typing import Any, Dict, List, Optional
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from functools import reduce
from jaxtyping import PyTree

# mean, count can be scalar or vector
LogTuple = namedtuple('LogTuple', ['mean', 'count'])

"""
type checks
"""

def _is_scalar(x):
    return isinstance(x, int) or isinstance(x, float) or (isinstance(x, jnp.ndarray) and len(x.shape) == 0) or (isinstance(x, np.ndarray) and len(x.shape) == 0)

def _is_vector(x):
    return (isinstance(x, jnp.ndarray) and len(x.shape) > 0) or (isinstance(x, np.ndarray) and len(x.shape) > 0)

def _is_leaf(x):
    return _is_vector(x) or _is_scalar(x) or isinstance(x, LogTuple)

"""
helper functions
"""

# convert jax objects to python list and float
def _un_jax_logs(logs):
    def _un_jax_log_f(x):
        if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
            if len(x.shape) == 0:
                return float(x.item())
            else:
                return list(map(float, x.tolist()))
        return x
    return jax.tree_util.tree_map(_un_jax_log_f, logs)

# average logs
def _reduce_elements(x):
    if isinstance(x, LogTuple):
        if _is_vector(x.mean):
            return jnp.nan_to_num((x.mean * x.count).sum() / x.count.sum(), nan=0, posinf=0, neginf=0)
        return x.mean
    if _is_vector(x):
        return x.mean()
    if _is_scalar(x):
        return x
    raise NotImplementedError

# concatenate logs
def _combine_elements(a, b):
    if _is_scalar(a):
        a = LogTuple(a, 1)
    if _is_scalar(b):
        b = LogTuple(b, 1)
    if isinstance(a, LogTuple) and isinstance(b, LogTuple):
        return LogTuple(jnp.nan_to_num((a.mean * a.count + b.mean * b.count) / (a.count + b.count), nan=0, posinf=0, neginf=0), a.count + b.count)
    if _is_vector(a) and _is_vector(b):
        return jnp.concatenate((a, b,), axis=0)
    raise NotImplementedError

"""
main log functions
"""

# combines a list of pytree logs into a single pytree of logs
def combine_logs(logs: List[PyTree], initial_log: Optional[PyTree]=None) -> PyTree:
    tree_def = jax.tree_util.tree_structure(logs[0], is_leaf=_is_leaf)
    flat_logs = list(zip(*[jax.tree_util.tree_flatten(log, is_leaf=_is_leaf)[0] for log in logs]))
    if initial_log is None:
        return jax.tree_util.tree_unflatten(tree_def, [reduce(_combine_elements, item) for item in flat_logs])
    return jax.tree_util.tree_unflatten(tree_def, [reduce(_combine_elements, item, initial_log) for item in flat_logs])

# averages logs and pulls them from jax array to python list and float
def pull_logs(logs: PyTree) -> Any:
    logs = jax.tree_util.tree_map(_reduce_elements, logs, is_leaf=_is_leaf)
    logs = jax.device_get(logs)
    logs = _un_jax_logs(logs)
    return logs

# add wrapper dict around logs
def label_logs(logs: Any, label: str, to_add: Dict[str, Any]) -> Dict[str, Any]:
    return {label: logs, **to_add}

# log with print and wandb
def log(logs: Any, use_wandb: bool) -> None:
    if use_wandb:
        wandb.log(logs)
    print(logs)