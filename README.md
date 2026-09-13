# muybridge

Morphology-robust velocity-tracking locomotion for the Unitree G1 in
MuJoCo MJX, trained with PPO. See [SPEC.md](SPEC.md) for the full brief,
stage criteria, and git workflow.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e .
```

The first import of MuJoCo Playground clones MuJoCo Menagerie (pinned commit)
into the package's `external_deps` directory. MJX runs on the JAX backend
(`impl="jax"`); nothing assumes CUDA or Metal.

## Layout

- `muybridge/model.py` — loads the Playground G1 scene through `MjSpec` and
  fuses the 14 arm joints into the torso at runtime. Arm links keep mass and
  inertia; the policy sees 15 actuators (12 leg + 3 waist). No XML is edited.
- `muybridge/bench.py` — Stage 0 random-action throughput benchmark.
- `muybridge/jax_cache.py` — persistent JAX compilation cache
  (`~/.cache/muybridge_jax`, override with `MUYBRIDGE_JAX_CACHE`).
- `reports/` — per-stage summaries and benchmark/eval outputs.

## Stage 0

```bash
python -m muybridge.bench --num-envs 64 256 512 --full-arms --out reports/stage0_throughput.json
```

## Stage 1

```bash
# train (TensorBoard scalars in runs/<run>/tb, checkpoints every 50 iterations)
python -m muybridge.train --run runs/stage1_a --num-envs 512 --num-timesteps 50_000_000
tensorboard --logdir runs

# fixed eval harness on the latest checkpoint (held-out seeds, 45-command grid, 20 s episodes)
python -m muybridge.eval --run runs/stage1_a --seeds 3

# browser 3D viewer with checkpoint hot-reload, manual velocity commands and overlays
python -m muybridge.viewer --run runs/stage1_a --port 8080
```

`docs/upstream_diff.md` lists every deliberate difference from the upstream
MuJoCo Playground G1 joystick task. Tests: `pytest tests`.
