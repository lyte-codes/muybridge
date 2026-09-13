"""Checkpoint directory helpers shared by eval and the viewer.

``load_policy`` re-implements Brax's loader because Brax 0.14.2 writes
``null`` kernel-initializer entries that its own ``load_config`` cannot read.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from brax.training import checkpoint as brax_checkpoint
from brax.training import networks as brax_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

CONFIG_FNAME = "ppo_network_config.json"
KERNEL_INIT_KEYS = ("policy_network_kernel_init_fn", "value_network_kernel_init_fn", "q_network_kernel_init_fn", "mean_kernel_init_fn")


def list_checkpoints(ckpt_dir: str) -> List[str]:
  if not os.path.isdir(ckpt_dir):
    return []
  out = []
  for name in os.listdir(ckpt_dir):
    path = os.path.join(ckpt_dir, name)
    if name.isdigit() and os.path.isdir(path) and os.path.exists(os.path.join(path, CONFIG_FNAME)):
      out.append(path)
  return sorted(out, key=lambda p: int(os.path.basename(p)))


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
  ckpts = list_checkpoints(ckpt_dir)
  return ckpts[-1] if ckpts else None


def checkpoint_step(path: str) -> int:
  return int(os.path.basename(os.path.normpath(path)))


def load_config(path: str) -> config_dict.ConfigDict:
  with open(os.path.join(path, CONFIG_FNAME)) as f:
    raw = json.load(f)
  kw = raw["network_factory_kwargs"]
  if isinstance(kw.get("activation"), str):
    kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
  for key in KERNEL_INIT_KEYS:
    if key in kw and kw[key] is not None:
      kw[key] = brax_networks.KERNEL_INITIALIZER[kw[key]]
  return config_dict.create(**raw)


def observation_sizes(path: str) -> Dict[str, int]:
  with open(os.path.join(path, CONFIG_FNAME)) as f:
    raw = json.load(f)
  obs = raw["observation_size"]
  if isinstance(obs, int):
    return {"state": obs}
  return {k: int(v["shape"][0]) if isinstance(v, dict) else int(v) for k, v in obs.items()}


def load_policy(path: str, deterministic: bool = True):
  path = os.path.abspath(path)
  config = load_config(path)
  params = ppo_checkpoint.load(path)
  network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
  return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)
