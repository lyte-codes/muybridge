"""Persistent JAX compilation cache so compiled MJX functions survive across processes."""

import os

import jax

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "muybridge_jax")


def enable_persistent_cache(cache_dir: str | None = None) -> str:
  cache_dir = cache_dir or os.environ.get("MUYBRIDGE_JAX_CACHE", DEFAULT_CACHE_DIR)
  os.makedirs(cache_dir, exist_ok=True)
  jax.config.update("jax_compilation_cache_dir", cache_dir)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
  return cache_dir
