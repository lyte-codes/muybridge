"""PPO training entry point.

    python -m muybridge.train --run runs/stage1_a --num-envs 256 --num-timesteps 50_000_000

Training scalars go to TensorBoard (``<run>/tb``) and ``<run>/metrics.jsonl``;
checkpoints go to ``<run>/checkpoints/<step>`` every ``--ckpt-every-iters``
PPO iterations. The 3D viewer (``muybridge.viewer``) polls that directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict

from brax.training.agents.ppo import train as ppo_train
import jax
from ml_collections import config_dict
from mujoco_playground._src import wrapper
import numpy as np
from tensorboardX import SummaryWriter

import muybridge  # noqa: F401
from muybridge import compat, envs, networks, ppo_losses
from muybridge.envs import g1_joystick

compat.install()
ppo_train.ppo_losses = ppo_losses


def ppo_config(args: argparse.Namespace) -> config_dict.ConfigDict:
  if args.num_envs % args.num_minibatches:
    raise ValueError("num_envs must be divisible by num_minibatches")
  return config_dict.create(
      num_timesteps=args.num_timesteps,
      num_envs=args.num_envs,
      unroll_length=args.unroll_length,
      num_minibatches=args.num_minibatches,
      batch_size=args.num_envs // args.num_minibatches,
      num_updates_per_batch=args.num_updates_per_batch,
      discounting=args.discounting,
      gae_lambda=0.95,
      learning_rate=args.learning_rate,
      entropy_cost=args.entropy_cost,
      clipping_epsilon=0.2,
      max_grad_norm=1.0,
      normalize_observations=True,
      reward_scaling=1.0,
      action_repeat=1,
      seed=args.seed,
      policy_hidden=list(networks.POLICY_HIDDEN),
      value_hidden=list(networks.VALUE_HIDDEN),
  )


def git_sha() -> str:
  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  except Exception:  # noqa: BLE001
    return "unknown"


def rename_metric(key: str) -> str:
  """Map Brax's metric names onto stable TensorBoard groups."""
  if key.startswith("episode/"):
    rest = key[len("episode/"):]
    if rest.startswith(("reward/", "track/", "term/", "gait/")):
      return rest[:-len("_per_step")] if rest.endswith("_per_step") else rest
    if rest in ("sum_reward", "length"):
      return f"episode/{rest}"
    if rest == "sps":
      return "throughput/logger_env_steps_per_s"
    return f"ppo/{rest}"
  if key.startswith("training/"):
    rest = key[len("training/"):]
    if rest == "sps":
      return "throughput/env_steps_per_s"
    if rest == "walltime":
      return "throughput/walltime_s"
    return f"ppo_epoch/{rest}"
  return key


class Logger:

  def __init__(self, run_dir: str, steps_per_iter: int, iters_per_epoch: int):
    self.writer = SummaryWriter(logdir=os.path.join(run_dir, "tb"))
    self.jsonl = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    self.steps_per_iter = steps_per_iter
    self.iters_per_epoch = iters_per_epoch
    self.last_walltime = None
    self.t_start = time.time()

  def __call__(self, step: int, metrics: Dict[str, Any]) -> None:
    scalars = {}
    for k, v in metrics.items():
      try:
        scalars[rename_metric(k)] = float(np.asarray(v))
      except (TypeError, ValueError):
        continue
    if "term/fall" in scalars and "term/self_collision" in scalars:
      scalars["term/timeout"] = max(0.0, 1.0 - scalars["term/fall"] - scalars["term/self_collision"])
    if "throughput/walltime_s" in scalars:
      wt = scalars["throughput/walltime_s"]
      if self.last_walltime is not None:
        scalars["throughput/wallclock_per_iter_s"] = (wt - self.last_walltime) / self.iters_per_epoch
      self.last_walltime = wt
    scalars["throughput/iteration"] = step / self.steps_per_iter
    for k, v in scalars.items():
      self.writer.add_scalar(k, v, step)
    self.writer.flush()
    record = {"step": step, "time": time.time() - self.t_start, **scalars}
    self.jsonl.write(json.dumps(record) + "\n")
    self.jsonl.flush()
    summary = {k: scalars[k] for k in ("episode/sum_reward", "episode/length", "track/lin_vel_err", "term/fall", "throughput/env_steps_per_s") if k in scalars}
    print(f"[step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)


def main(argv=None) -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, help="run directory")
  p.add_argument("--env", default="G1JoystickFlat")
  p.add_argument("--env-override", action="append", default=[], help="key=value on the env config, e.g. noise_config.level=0.5")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--num-timesteps", type=int, default=50_000_000)
  p.add_argument("--unroll-length", type=int, default=20)
  p.add_argument("--num-minibatches", type=int, default=4)
  p.add_argument("--num-updates-per-batch", type=int, default=4)
  p.add_argument("--discounting", type=float, default=0.97)
  p.add_argument("--learning-rate", type=float, default=3e-4)
  p.add_argument("--entropy-cost", type=float, default=0.005)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--ckpt-every-iters", type=int, default=50)
  p.add_argument("--restore", default=None, help="checkpoint directory to restore params from")
  p.add_argument("--brax-evals", action="store_true", help="also run Brax's built-in evaluator each epoch")
  args = p.parse_args(argv)

  overrides = {}
  for item in args.env_override:
    k, v = item.split("=", 1)
    overrides[k] = json.loads(v) if v[0] in "[{-0123456789tfn" else v

  env_cfg = envs.default_config(args.env)
  env = envs.make(args.env, env_cfg, overrides)
  cfg = ppo_config(args)
  steps_per_iter = cfg.batch_size * cfg.unroll_length * cfg.num_minibatches * cfg.action_repeat
  num_evals = max(1, math.ceil(cfg.num_timesteps / (args.ckpt_every_iters * steps_per_iter)))
  iters_per_epoch = math.ceil(cfg.num_timesteps / (num_evals * steps_per_iter))

  os.makedirs(args.run, exist_ok=True)
  ckpt_dir = os.path.abspath(os.path.join(args.run, "checkpoints"))
  run_config = {
      "env": args.env,
      "env_version": g1_joystick.ENV_VERSION,
      "env_config": env._config.to_dict(),
      "ppo": cfg.to_dict(),
      "steps_per_iter": steps_per_iter,
      "num_evals": num_evals,
      "iters_per_epoch": iters_per_epoch,
      "git_sha": git_sha(),
      "argv": sys.argv,
      "jax_backend": jax.default_backend(),
      "started": datetime.datetime.now().isoformat(timespec="seconds"),
  }
  with open(os.path.join(args.run, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2, default=str)
  print(f"steps/iter={steps_per_iter} iters/epoch={iters_per_epoch} num_evals={num_evals} ckpt_dir={ckpt_dir}", flush=True)

  logger = Logger(args.run, steps_per_iter, iters_per_epoch)
  ppo_train.train(
      environment=env,
      num_timesteps=cfg.num_timesteps,
      num_envs=cfg.num_envs,
      episode_length=env_cfg.episode_length,
      action_repeat=cfg.action_repeat,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      learning_rate=cfg.learning_rate,
      entropy_cost=cfg.entropy_cost,
      discounting=cfg.discounting,
      unroll_length=cfg.unroll_length,
      batch_size=cfg.batch_size,
      num_minibatches=cfg.num_minibatches,
      num_updates_per_batch=cfg.num_updates_per_batch,
      normalize_observations=cfg.normalize_observations,
      reward_scaling=cfg.reward_scaling,
      clipping_epsilon=cfg.clipping_epsilon,
      gae_lambda=cfg.gae_lambda,
      max_grad_norm=cfg.max_grad_norm,
      network_factory=networks.make_network_factory(cfg.policy_hidden, cfg.value_hidden),
      seed=cfg.seed,
      num_evals=num_evals,
      num_eval_envs=128,
      run_evals=args.brax_evals,
      log_training_metrics=True,
      training_metrics_steps=steps_per_iter,
      progress_fn=logger,
      save_checkpoint_path=ckpt_dir,
      restore_checkpoint_path=args.restore,
  )
  print("training finished", flush=True)


if __name__ == "__main__":
  main()
