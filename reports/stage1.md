# Stage 1 — Flat-ground velocity tracking

Status: **PR #2 draft, run g complete (20.4M samples)**. Pass criteria not yet met; see "Where it
stands" below. Everything here was produced on a 4-core CPU sandbox at
roughly 700–900 env steps/s (512 envs), i.e. about 3 % of the sample budget
the upstream G1 task uses. Numbers should be re-established on a GPU.

## What changed (vs. Stage 0)

- `muybridge/envs/g1_joystick.py` — velocity-tracking env forked from the
  Playground G1 joystick task on the fused-arm 15-actuator model. Policy obs
  = gyro, projected gravity, command, joint pos/vel, previous action, stacked
  over a 10-frame history (540 dims). Critic gets a privileged state. Every
  difference from upstream is tabulated in `docs/upstream_diff.md`.
- `muybridge/train.py` — Brax PPO, TensorBoard + JSONL logging (each reward
  term, tracking errors, termination causes, PPO diagnostics incl. clip
  fraction and explained variance via a forked loss, throughput), checkpoints
  every 50 iterations, warm start from any checkpoint.
- `muybridge/eval.py` — fixed harness: held-out seeds, 45-command grid,
  20 s episodes, tracking error / completion / time-to-failure / cost of
  transport / smoothness, versioned JSON + `history.csv`.
- `muybridge/viewer.py` — Viser 3D viewer with hot-reload, checkpoint
  selector, play/pause/step/reset, command sliders, follow camera and all
  spec overlays (verified by headless screenshots).
- `muybridge/compat.py` — shim for `jax.device_put_replicated` (removed in
  JAX 0.10, still used by Brax 0.14.2).

## Env version history (all within this branch)

| version | change | evidence that motivated it |
|---|---|---|
| 1.0.0 | initial fork | — |
| 1.0.1 | `alive` +1.0, `action_rate` −0.05 → −0.01 | per-step reward was net negative, so terminating was optimal: every episode ended by deliberate foot-foot contact at 0.9 s |
| 1.0.2 | reset joint scale U(0.5,1.5) → U(0.8,1.2), base vel ±0.5 → ±0.2 (configurable) | episode length stuck at ~45 steps for 1.8M samples; a zero-action robot topples in ~1.2 s, so upstream's half-fallen starts left no time to learn balance |
| 1.0.3 | self-collision is a −1.0/step penalty, termination only on fall (+ pelvis height < 0.35 m) | 93 % of episodes ended by foot-foot contact at 3M samples; spec lists self-collision under penalties and termination under falls; feet-only floor collision let a kneeling robot sink through the ground |
| 1.0.4 | tracking weights 1.5 / 1.0, `alive` 0.5, air-time threshold 0.1 s | after standing was learned the tracking reward stayed flat; steps < 0.2 s were penalised |
| 1.0.5 | true foot slip (foot xy speed × contact), `feet_air_time` 5.0 masked to non-zero commands, sigma back to 0.25, `alive` 0.25 | policy parked in a standing optimum: upstream's slip proxy charges body speed during stance (i.e. walking), sigma 0.5 paid 61 % of full tracking reward for standing still |
| (option) | `gait_phase.enable` adds a 4-dim phase clock + upstream `feet_phase` reward, **off by default** | upstream's gait-discovery mechanism; changes the interface contract, so it is an ablation pending sign-off |

## Runs (chronological, each warm-started from the previous where noted)

| run | env | samples | outcome |
|---|---|---|---|
| a | 1.0.0 | 1.8M | fall rate → 0 but 99 % self-collision terminations at 0.9 s (reward hack) |
| b | 1.0.1 | 1.2M (container reclaimed once) | ~45-step plateau |
| c | 1.0.2 | 3.0M | length 111, 93 % foot-foot terminations |
| d | 1.0.3, warm from c | 3.7M | length 430, stands reliably, does not walk |
| e | 1.0.4, warm from d | 1.0M | stands perfectly (0.04 m/s error at zero command), ignores forward commands |
| f | 1.0.5, warm from e | see below | walks |

## Eval harness (run g, env 1.0.5, 45 commands × 3 held-out seeds, 20 s episodes)

Cumulative samples across the warm-start chain c→d→e→f→g are ~19M + the run-g step shown.

| run g step | completion | falls / 135 | lin err mean (m/s) | lin err max | ang err mean (rad/s) | ang err max | CoT | action smoothness | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1,044,480 | 97.8 % | 3 | 0.226 | 0.954 | 0.334 | 1.864 | 9.8 | 0.191 | no |
| 2,088,960 | 96.3 % | 5 | 0.201 | 0.878 | 0.327 | 2.595 | 11.8 | 0.191 | no |
| 2,611,200 | 84.4 % | 21 | 0.214 | 1.605 | 0.344 | 2.825 | 10.0 | 0.192 | no |
| 3,133,440 | 88.1 % | 16 | 0.214 | 1.249 | 0.327 | 2.679 | 10.5 | 0.190 | no |
| 3,655,680 | 89.6 % | 14 | 0.222 | 2.713 | 0.339 | 3.783 | 8.7 | 0.190 | no |
| 4,700,160 | 91.1 % | 12 | 0.196 | 1.345 | 0.282 | 1.646 | 10.5 | 0.189 | no |
| 5,222,400 | 86.7 % | 18 | 0.224 | 1.651 | 0.318 | 1.930 | 9.1 | 0.190 | no |
| 5,744,640 | 87.4 % | 17 | 0.192 | 1.197 | 0.279 | 1.570 | 9.9 | 0.188 | no |
| 6,266,880 | 84.4 % | 21 | 0.226 | 1.856 | 0.308 | 1.591 | 8.7 | 0.191 | no |
| 6,789,120 | 85.9 % | 19 | 0.193 | 1.358 | 0.300 | 1.999 | 8.9 | 0.189 | no |
| 7,311,360 | 85.9 % | 19 | 0.204 | 2.031 | 0.314 | 2.576 | 10.2 | 0.185 | no |
| 7,833,600 | 86.7 % | 18 | 0.198 | 1.277 | 0.326 | 3.405 | 11.4 | 0.187 | no |
| 8,355,840 | 88.1 % | 16 | 0.196 | 2.055 | 0.287 | 1.482 | 11.7 | 0.190 | no |
| 8,878,080 | 88.9 % | 15 | 0.213 | 1.551 | 0.308 | 1.865 | 10.9 | 0.191 | no |
| 9,400,320 | 91.9 % | 11 | 0.177 | 0.507 | 0.269 | 0.825 | 11.0 | 0.186 | no |
| 9,922,560 | 87.4 % | 17 | 0.181 | 0.677 | 0.283 | 1.211 | 14.3 | 0.189 | no |
| 10,444,800 | 89.6 % | 14 | 0.178 | 1.024 | 0.268 | 1.585 | 12.1 | 0.190 | no |
| 10,967,040 | 91.9 % | 11 | 0.188 | 2.001 | 0.278 | 1.597 | 11.8 | 0.191 | no |
| 11,489,280 | 89.6 % | 14 | 0.184 | 1.529 | 0.275 | 1.755 | 11.0 | 0.191 | no |
| 12,533,760 | 85.9 % | 19 | 0.182 | 1.185 | 0.280 | 2.384 | 11.3 | 0.190 | no |
| 13,056,000 | 83.7 % | 22 | 0.185 | 1.255 | 0.297 | 2.645 | 12.1 | 0.190 | no |
| 14,100,480 | 94.1 % | 8 | 0.162 | 0.468 | 0.239 | 0.789 | 14.3 | 0.187 | no |
| 14,622,720 | 94.8 % | 7 | 0.156 | 0.405 | 0.236 | 0.711 | 10.7 | 0.188 | no |
| 15,144,960 | 94.1 % | 8 | 0.159 | 0.559 | 0.240 | 0.841 | 16.0 | 0.187 | no |
| 15,667,200 | 90.4 % | 13 | 0.171 | 1.487 | 0.250 | 1.310 | 14.0 | 0.188 | no |
| 16,189,440 | 89.6 % | 14 | 0.173 | 0.891 | 0.257 | 1.496 | 10.7 | 0.189 | no |
| 16,711,680 | 91.1 % | 12 | 0.196 | 1.991 | 0.284 | 4.615 | 14.3 | 0.188 | no |
| 17,233,920 | 90.4 % | 13 | 0.189 | 1.220 | 0.263 | 1.800 | 14.2 | 0.188 | no |
| 17,756,160 | 92.6 % | 10 | 0.167 | 0.424 | 0.216 | 0.884 | 12.8 | 0.187 | no |
| 18,278,400 | 91.9 % | 11 | 0.192 | 1.750 | 0.261 | 2.457 | 11.5 | 0.185 | no |
| 18,800,640 | 90.4 % | 13 | 0.178 | 2.258 | 0.262 | 3.175 | 13.8 | 0.188 | no |
| 19,322,880 | 89.6 % | 14 | 0.184 | 1.562 | 0.254 | 1.762 | 11.5 | 0.186 | no |
| 19,845,120 | 93.3 % | 9 | 0.185 | 2.270 | 0.269 | 4.582 | 11.8 | 0.186 | no |
| 20,367,360 | 90.4 % | 13 | 0.163 | 0.812 | 0.223 | 1.151 | 12.6 | 0.184 | no |

Breakdown of the 1,044,480 checkpoint by command type (mean lin / ang error):
zero command 0.04 / 0.09, forward-only 0.11 / 0.22, lateral-only 0.18 / 0.27,
yaw-only 0.08 / 0.40, combined (≥2 axes) 0.25 / 0.35. 18 of 135 episodes meet
the 0.1 m/s linear bar and 3 the 0.1 rad/s yaw bar. Yaw is the weakest axis;
the worst grid points are the (vx, vy, wz) corners with all three non-zero.

## Where it stands

- The pipeline learns: from a policy that fell in 0.9 s to one that stands,
  walks forward/backward at ~0.1 m/s error and completes 96–98 % of 20 s
  held-out episodes at its best checkpoint.
- **Pass criteria are not met** (0.1 m/s at every grid point, zero falls in
  100 episodes). The gap is largest on yaw and on combined commands.
- Later checkpoints of run g regress on completion (84–88 %) while the
  training reward keeps rising, which is exactly the training-curve vs.
  harness disagreement the spec warns about. Best completion by harness:
  `runs/stage1_cpu_g/checkpoints/000001044480` (97.8 %); best tracking:
  the final checkpoint `000020367360` (0.163 m/s / 0.223 rad/s, 90.4 %).
- Training-time fall rate at command re-sampling (every 10 s) is still
  ~20 %, so transitions between commands are where the falls come from.


## Regressions

None against Stage 0 (no policy existed). Within the stage, each env
revision invalidated the previous checkpoints' reward scale; warm starts
carried the policy across.

## Open questions

- **Gait phase clock.** Upstream G1 discovers its gait through a phase clock
  in the observation plus a `feet_phase` reward. The spec's observation list
  excludes it and it changes the interface contract, so the baseline runs
  without it. It is implemented behind `gait_phase.enable` for an A/B; the
  decision needs sign-off.
- **Compute.** Everything above is ~10M cumulative samples on CPU. Upstream
  trains this task for 200M. The GPU run is the real Stage 1 run; the CPU
  runs establish that the pipeline learns and where the reward shaping
  needed fixing.
- **Pass-criterion strictness.** "Within 0.1 m/s across the grid" is
  interpreted as the per-command mean error after a 2 s settle, at every one
  of the 45 grid points including the 1.0 m/s corners.
