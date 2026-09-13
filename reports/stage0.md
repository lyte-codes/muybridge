# Stage 0 — Environment

## What changed

- MuJoCo 3.13 / MJX / JAX 0.10 / Brax 0.14.2 / MuJoCo Playground 0.2.0 pinned;
  Menagerie cloned at Playground's pinned commit. Backend is `impl="jax"`
  (upstream Playground now defaults the G1 config to `impl="warp"`, which is
  CUDA-only).
- `muybridge/model.py` loads the upstream G1 scene through `MjSpec` and fuses
  the 14 arm joints into the torso at the `knees_bent` pose. Verified: total
  mass identical (33.3411 kg), palm site positions and torso/whole-body CoM
  identical to the 29-DoF model at that pose to 1e-5 m. The policy-facing
  model has 15 position actuators (12 leg + 3 waist), nq=22, nv=21.
- `muybridge/bench.py` random-action rollout benchmark.
- Persistent JAX compilation cache (`~/.cache/muybridge_jax`).

## Throughput

Machine: 4-core x86_64 Linux sandbox, JAX CPU backend (no GPU). Control step
= 10 physics substeps (ctrl_dt 0.02 s, sim_dt 0.002 s). 50 control steps per
env per timed call, compile excluded. Raw data: `stage0_throughput.json`.

| envs | model | compile + first call | ctrl steps/s | physics steps/s |
|-----:|-------|---------------------:|-------------:|----------------:|
|   64 | fused arms (nu=15) | 24.9 s | 164 | 1 638 |
|  256 | fused arms (nu=15) | 32.6 s | 430 | 4 304 |
|  512 | fused arms (nu=15) | 43.1 s | 630 | 6 297 |
|   64 | full 29-DoF        | 25.9 s | 139 | 1 387 |
|  256 | full 29-DoF        | 39.6 s | 344 | 3 441 |
|  512 | full 29-DoF        | 54.4 s | 499 | 4 991 |

No NaNs in any rollout. Fusing the arms is 20–26 % faster than simulating
them, which is the expected benefit of the arm-as-torso-load design.

## Regressions

None (first stage).

## Open questions

- **CPU throughput is low for training.** At ~630 control steps/s, 100 M env
  steps (the upstream G1 PPO budget) would take ~44 h on this box. The M2 Air
  will be in the same regime because MJX has no Metal contact path. Plan:
  local runs are smoke tests at 64–256 envs; the real Stage 1 run goes to a
  rented NVIDIA GPU. Numbers should be re-measured on the M2 and on the GPU
  and appended here.
- Arm fusion is baked at the `knees_bent` arm pose. Stage 4's payload/CoM
  randomization will move the effective torso load anyway; if a different
  fixed arm pose is preferred, it is a one-line change in `load_g1`.
