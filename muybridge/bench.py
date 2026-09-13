"""Stage 0 throughput benchmark: random-action rollouts of the G1 in MJX.

Measures physics steps/sec and control steps/sec for batches of parallel envs.
Compile time is reported separately and excluded from the throughput numbers.

    python -m muybridge.bench --num-envs 64 256 512 --out reports/stage0_throughput.json
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import time
from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

import muybridge  # noqa: F401  (enables the persistent JAX cache)
from muybridge.model import load_g1

CTRL_DT = 0.02
SIM_DT = 0.002


def make_rollout_fn(mx: mjx.Model, n_substeps: int, num_envs: int, num_steps: int):
  ctrl_lo = mx.actuator_ctrlrange[:, 0]
  ctrl_hi = mx.actuator_ctrlrange[:, 1]

  def control_step(data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
    def substep(d, _):
      return mjx.step(mx, d.replace(ctrl=ctrl)), None

    return jax.lax.scan(substep, data, None, n_substeps)[0]

  v_control_step = jax.vmap(control_step)

  @jax.jit
  def rollout(data: mjx.Data, rng: jax.Array):
    def body(carry, _):
      data, rng = carry
      rng, key = jax.random.split(rng)
      ctrl = jax.random.uniform(key, (num_envs, mx.nu), minval=ctrl_lo, maxval=ctrl_hi)
      return (v_control_step(data, ctrl), rng), None

    (data, rng), _ = jax.lax.scan(body, (data, rng), None, num_steps)
    return data, rng

  return rollout


def batched_initial_data(mj_model: mujoco.MjModel, num_envs: int, key_name: str) -> mjx.Data:
  data = mjx.make_data(mj_model, impl="jax")
  data = data.replace(qpos=jp.array(mj_model.key(key_name).qpos))
  return jax.tree.map(lambda x: jp.broadcast_to(x, (num_envs,) + x.shape), data)


def benchmark(num_envs: int, num_steps: int, fuse_arms: bool, terrain: str) -> Dict[str, Any]:
  g1 = load_g1(terrain=terrain, fuse_arms=fuse_arms, sim_dt=SIM_DT)
  mx = mjx.put_model(g1.mj_model, impl="jax")
  n_substeps = int(round(CTRL_DT / SIM_DT))
  rollout = make_rollout_fn(mx, n_substeps, num_envs, num_steps)
  data = batched_initial_data(g1.mj_model, num_envs, "knees_bent")
  rng = jax.random.PRNGKey(0)

  t0 = time.perf_counter()
  data, rng = rollout(data, rng)
  jax.block_until_ready(data)
  compile_and_first = time.perf_counter() - t0

  t0 = time.perf_counter()
  data, rng = rollout(data, rng)
  jax.block_until_ready(data)
  elapsed = time.perf_counter() - t0

  ctrl_steps = num_envs * num_steps
  physics_steps = ctrl_steps * n_substeps
  qpos = np.asarray(data.qpos)
  return {
      "num_envs": num_envs,
      "num_ctrl_steps_per_env": num_steps,
      "n_substeps": n_substeps,
      "fuse_arms": fuse_arms,
      "terrain": terrain,
      "nq": int(g1.mj_model.nq),
      "nv": int(g1.mj_model.nv),
      "nu": int(g1.mj_model.nu),
      "compile_plus_first_call_s": compile_and_first,
      "timed_rollout_s": elapsed,
      "ctrl_steps_per_s": ctrl_steps / elapsed,
      "physics_steps_per_s": physics_steps / elapsed,
      "any_nan": bool(np.isnan(qpos).any()),
      "mean_base_height": float(qpos[:, 2].mean()),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 256, 512])
  parser.add_argument("--steps", type=int, default=50, help="control steps per env per timed call")
  parser.add_argument("--terrain", default="flat", choices=["flat", "rough"])
  parser.add_argument("--full-arms", action="store_true", help="also benchmark the 29-DoF model")
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  results = {
      "jax_version": jax.__version__,
      "mujoco_version": mujoco.__version__,
      "backend": jax.default_backend(),
      "devices": [str(d) for d in jax.devices()],
      "platform": platform.platform(),
      "machine": platform.machine(),
      "cpu_count": jax.device_count(),
      "runs": [],
  }
  variants = [True] + ([False] if args.full_arms else [])
  for fuse_arms in variants:
    for n in args.num_envs:
      r = benchmark(n, args.steps, fuse_arms, args.terrain)
      results["runs"].append(r)
      print(
          f"envs={n:4d} fused_arms={fuse_arms!s:5} nu={r['nu']:2d} "
          f"compile+first={r['compile_plus_first_call_s']:6.1f}s "
          f"ctrl_steps/s={r['ctrl_steps_per_s']:9.0f} physics_steps/s={r['physics_steps_per_s']:10.0f} "
          f"nan={r['any_nan']}",
          flush=True,
      )
  if args.out:
    with open(args.out, "w") as f:
      json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
  main()
