"""Compatibility shims between pinned library versions.

Brax 0.14.2 calls ``jax.device_put_replicated``, which JAX >= 0.10 removed.
This installs the drop-in replacement from JAX's pmap migration guide: the
value is stacked along a new leading device axis and sharded across devices,
which is what Brax's ``pmap``-based training loop expects.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _device_put_replicated(x, devices):
  devices = list(devices)
  mesh = jax.sharding.Mesh(np.array(devices), ("d",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
  stacked = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (len(devices),) + jnp.shape(a)), x)
  return jax.device_put(stacked, sharding)


def install() -> None:
  if not hasattr(jax, "device_put_replicated") or _is_removed(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated


def _is_removed(module, name) -> bool:
  try:
    getattr(module, name)
    return False
  except AttributeError:
    return True
