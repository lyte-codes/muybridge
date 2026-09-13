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
