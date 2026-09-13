"""Fixed evaluation harness, reusable across all stages.

    python -m muybridge.eval --run runs/stage1_a            # latest checkpoint
    python -m muybridge.eval --ckpt runs/stage1_a/checkpoints/000001000000

Held-out seeds (never used by training, which seeds from 0), a fixed command
grid, 20 s episodes. Writes ``<run>/eval/<step>.json`` and appends one row to
``<run>/eval/history.csv`` so checkpoints can be compared.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jp
import numpy as np

import muybridge  # noqa: F401
from muybridge import checkpoints, envs
from muybridge.envs import g1_joystick

HARNESS_VERSION = "1.0.0"
EVAL_SEED_BASE = 10_000
COMMAND_GRID = {
    "vx": (-1.0, -0.5, 0.0, 0.5, 1.0),
    "vy": (-0.5, 0.0, 0.5),
    "wz": (-1.0, 0.0, 1.0),
}
LIN_ERR_PASS = 0.1  # m/s
ANG_ERR_PASS = 0.1  # rad/s, interpreted from the spec's "within 0.1" for yaw
SETTLE_TIME = 2.0  # s, excluded from tracking-error means


def command_grid() -> np.ndarray:
  return np.array(list(itertools.product(COMMAND_GRID["vx"], COMMAND_GRID["vy"], COMMAND_GRID["wz"])), dtype=np.float32)


def make_eval_env(env_name: str, env_overrides: Optional[Dict[str, Any]] = None, noise_level: float = 1.0):
  overrides = {"command_config.resample_steps": 10**9, "noise_config.level": noise_level}
  overrides.update(env_overrides or {})
  return envs.make(env_name, None, overrides)


def rollout(env, policy, commands: np.ndarray, seeds: np.ndarray, episode_length: int) -> Dict[str, np.ndarray]:
  """Batched rollout of one episode per (command, seed) pair. Returns per-step arrays (T, N)."""
  n = commands.shape[0]
  keys = jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds))
  state = jax.vmap(env.reset)(keys)
  state.info["command"] = jp.asarray(commands)
  frame = jax.vmap(lambda d, i: env._frame(d, i, noisy=False))(state.data, state.info)
  state.info["obs_history"] = jp.tile(frame[:, None], (1, env._config.history_len, 1))
  state = state.replace(obs=jax.vmap(env._get_obs)(state.data, state.info, jax.vmap(env.feet_contact)(state.data), frame))
  mass = float(env.mj_model.body_subtreemass[1])

  def step(carry, _):
    state, alive, key = carry
    key, sub = jax.random.split(key)
    act, _ = policy(state.obs, jax.random.split(sub, n))
    nstate = jax.vmap(env.step)(state, act)
    data = nstate.data
    linvel = jax.vmap(env.get_local_linvel)(data)
    gyro = jax.vmap(env.get_gyro)(data)
    cmd = nstate.info["command"]
    rec = {
        "alive": alive,
        "done": nstate.done,
        "fall": nstate.metrics["term/fall"],
        "self_collision": nstate.metrics["gait/self_collision_per_step"],
        "lin_err": jp.linalg.norm(cmd[:, :2] - linvel[:, :2], axis=-1),
        "ang_err": jp.abs(cmd[:, 2] - gyro[:, 2]),
        "power": jp.sum(jp.abs(data.actuator_force * data.qvel[:, 6:]), axis=-1),
        "action_delta": jp.mean(jp.abs(act - state.info["last_act"]), axis=-1),
        "xy": data.qpos[:, :2],
        "torque_abs": jp.sum(jp.abs(data.actuator_force), axis=-1),
    }
    new_alive = alive * (1.0 - nstate.done)
    return (nstate, new_alive, key), rec

  scan = jax.jit(lambda s, a, k: jax.lax.scan(step, (s, a, k), None, episode_length))
  (_, _, _), recs = scan(state, jp.ones(n), jax.random.PRNGKey(EVAL_SEED_BASE + 1))
  recs = {k: np.asarray(v) for k, v in recs.items()}
  recs["mass"] = mass
  return recs


def _nan_to_none(x) -> Optional[float]:
  x = float(x)
  return None if np.isnan(x) else x


def summarize(recs: Dict[str, np.ndarray], commands: np.ndarray, seeds: np.ndarray, dt: float, episode_length: int) -> Dict[str, Any]:
  alive = recs["alive"]  # (T, N): 1 while the episode is still running at step t
  T, n = alive.shape
  steps_alive = alive.sum(0)
  completed = steps_alive >= T
  failed = ~completed
  t = np.arange(T) * dt
  settle = (t >= SETTLE_TIME)[:, None]
  w = alive * settle
  has_data = w.sum(0) > 0
  lin_err = np.where(has_data, (recs["lin_err"] * w).sum(0) / np.maximum(w.sum(0), 1), np.nan)
  ang_err = np.where(has_data, (recs["ang_err"] * w).sum(0) / np.maximum(w.sum(0), 1), np.nan)
  smooth = (recs["action_delta"] * alive).sum(0) / np.maximum(steps_alive, 1)
  energy = (recs["power"] * alive).sum(0) * dt
  xy0 = recs["xy"][0]
  last_idx = np.clip(steps_alive.astype(int) - 1, 0, T - 1)
  xy_end = recs["xy"][last_idx, np.arange(n)]
  dist = np.linalg.norm(xy_end - xy0, axis=-1)
  moving = np.linalg.norm(commands[:, :2], axis=-1) > 0.1
  cot = np.where(moving & (dist > 0.05), energy / (recs["mass"] * 9.81 * np.maximum(dist, 1e-6)), np.nan)
  fell = ((recs["fall"] * alive).sum(0) > 0)
  collided = ((recs["self_collision"] * alive).sum(0) > 0)  # any self-contact during the episode (non-terminal)
  ttf = steps_alive * dt

  per_command: List[Dict[str, Any]] = []
  for i in range(n):
    per_command.append({
        "vx": float(commands[i, 0]), "vy": float(commands[i, 1]), "wz": float(commands[i, 2]), "seed": int(seeds[i]),
        "completed": bool(completed[i]), "time_alive_s": float(ttf[i]),
        "fall": bool(fell[i]), "self_collision": bool(collided[i]),
        "lin_vel_err": None if np.isnan(lin_err[i]) else float(lin_err[i]),
        "ang_vel_err": None if np.isnan(ang_err[i]) else float(ang_err[i]),
        "cost_of_transport": None if np.isnan(cot[i]) else float(cot[i]),
        "action_smoothness": float(smooth[i]), "distance_m": float(dist[i]),
    })
  ok = completed
  summary = {
      "episodes": int(n),
      "completion_rate": float(completed.mean()),
      "falls": int(fell.sum()),
      "self_collisions": int(collided.sum()),
      "mean_time_to_failure_s": float(ttf[failed].mean()) if failed.any() else None,
      "mean_episode_time_s": float(ttf.mean()),
      "episodes_with_tracking_data": int(has_data.sum()),
      "lin_vel_err_mean": _nan_to_none(np.nanmean(lin_err)) if has_data.any() else None,
      "lin_vel_err_max": _nan_to_none(np.nanmax(lin_err)) if has_data.any() else None,
      "lin_vel_err_mean_completed": _nan_to_none(np.nanmean(lin_err[ok])) if ok.any() else None,
      "ang_vel_err_mean": _nan_to_none(np.nanmean(ang_err)) if has_data.any() else None,
      "ang_vel_err_max": _nan_to_none(np.nanmax(ang_err)) if has_data.any() else None,
      "cost_of_transport_mean": float(np.nanmean(cot)) if np.isfinite(cot).any() else None,
      "action_smoothness_mean": float(smooth.mean()),
  }
  summary["pass_stage1"] = bool(
      summary["completion_rate"] == 1.0 and n >= 100 and has_data.all()
      and summary["lin_vel_err_max"] <= LIN_ERR_PASS and summary["ang_vel_err_max"] <= ANG_ERR_PASS
  )
  return {"summary": summary, "per_command": per_command}


def evaluate(ckpt: str, env_name: str, seeds_per_command: int, episode_length: int, noise_level: float,
             env_overrides: Optional[Dict[str, Any]] = None, commands: Optional[np.ndarray] = None) -> Dict[str, Any]:
  env = make_eval_env(env_name, env_overrides, noise_level)
  policy = checkpoints.load_policy(ckpt, deterministic=True)
  grid = command_grid() if commands is None else commands
  cmds = np.repeat(grid, seeds_per_command, axis=0)
  seeds = np.array([EVAL_SEED_BASE + i for i in range(cmds.shape[0])])
  t0 = time.time()
  recs = rollout(env, policy, cmds, seeds, episode_length)
  result = summarize(recs, cmds, seeds, env.dt, episode_length)
  result["meta"] = {
      "harness_version": HARNESS_VERSION,
      "env": env_name,
      "env_version": env._config.env_version,
      "checkpoint": os.path.abspath(ckpt),
      "checkpoint_step": checkpoints.checkpoint_step(ckpt),
      "seeds_per_command": seeds_per_command,
      "episode_length": episode_length,
      "episode_seconds": episode_length * env.dt,
      "noise_level": noise_level,
      "env_overrides": env_overrides or {},
      "eval_wallclock_s": time.time() - t0,
      "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
  }
  return result


def write_result(result: Dict[str, Any], out_dir: str) -> str:
  os.makedirs(out_dir, exist_ok=True)
  step = result["meta"]["checkpoint_step"]
  path = os.path.join(out_dir, f"{step:012d}.json")
  with open(path, "w") as f:
    json.dump(result, f, indent=2)
  hist = os.path.join(out_dir, "history.csv")
  row = {"checkpoint_step": step, **{k: v for k, v in result["summary"].items()}, "timestamp": result["meta"]["timestamp"]}
  new = not os.path.exists(hist)
  with open(hist, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new:
      w.writeheader()
    w.writerow(row)
  return path


def main(argv=None) -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", default=None)
  p.add_argument("--ckpt", default=None)
  p.add_argument("--env", default="G1JoystickFlat")
  p.add_argument("--seeds", type=int, default=3, help="episodes per grid command (45 commands)")
  p.add_argument("--episode-length", type=int, default=1000)
  p.add_argument("--noise-level", type=float, default=1.0)
  p.add_argument("--out", default=None)
  p.add_argument("--quick", action="store_true", help="smoke test: 3 commands, 1 seed, 100 steps; never a pass")
  args = p.parse_args(argv)
  if args.ckpt is None:
    if args.run is None:
      p.error("--run or --ckpt required")
    args.ckpt = checkpoints.latest_checkpoint(os.path.join(args.run, "checkpoints"))
    if args.ckpt is None:
      p.error("no checkpoints found")
  out_dir = args.out or os.path.join(args.run or os.path.dirname(os.path.dirname(args.ckpt)), "eval")
  commands = None
  if args.quick:
    commands = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    args.seeds, args.episode_length = 1, 100
  result = evaluate(args.ckpt, args.env, args.seeds, args.episode_length, args.noise_level, commands=commands)
  if args.quick:
    result["summary"]["pass_stage1"] = False
    out_dir = os.path.join(out_dir, "quick")
  path = write_result(result, out_dir)
  s = result["summary"]
  print(json.dumps(s, indent=2))
  print("wrote", path)


if __name__ == "__main__":
  main()
