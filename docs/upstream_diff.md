# Differences from upstream MuJoCo Playground

Upstream reference: `mujoco_playground._src.locomotion.g1.joystick` and
`g1/base.py`, Playground 0.2.0 (MuJoCo 3.13, Brax 0.14.2). Our fork lives in
`muybridge/envs/g1_joystick.py`. Nothing under `site-packages` is modified.

| Area | Upstream | muybridge | Why |
|---|---|---|---|
| Model loading | `MjModel.from_xml_string` of the feet-only scene, 29 actuators | `MjSpec` load, 14 arm joints + actuators deleted at runtime, arm links fused to torso at the `knees_bent` arm pose | Interface contract: policy controls legs + waist only; arms are an unmodeled torso load. No XML edits. |
| MJX backend | `impl="warp"` (CUDA) | `impl="jax"` (config value) | Must run on CPU / Apple silicon; GPU optional. |
| Policy observation | linvel, gyro, gravity, cmd, joint pos/vel (29), last action (29), gait phase clock — single frame | gyro, gravity, cmd, joint pos/vel (15), last action (15) — **stacked history of 10 frames** (540 dims) | Spec observation set; history so Stage 3 can infer body properties. Local linear velocity dropped (not in spec, hard to measure on hardware). Phase clock dropped (see open questions). |
| Critic observation | privileged state | privileged state incl. history, clean frame, linvel, angvel, accel, height, torques, contacts, foot vel, air time | Asymmetric actor-critic kept; the policy never sees it. |
| Action | 29 joint targets, `default + 0.5 * a` | 15 joint targets, same mapping | Interface contract. |
| Reward set | includes `feet_phase`, `feet_clearance`, `feet_height`, `contact_force`, `base_height`, `energy`, `alive` (weight 0) | phase/clearance/height/contact-force/base-height/energy removed; **added** `waist_deviation`; `alive` weight 1.0; nonzero weights on `lin_vel_z`, `torques`, `action_rate`, `dof_acc` (upstream 0) | Spec reward set + waist penalty. Phase-based feet rewards need the phase clock. `alive` > 0 keeps per-step reward positive so early termination is never the optimum (env 1.0.1: without it every episode ended by deliberate foot-foot contact at 0.9 s). |
| Termination | torso upvector z < 0, foot-foot / foot-shin contact, NaN | same plus pelvis height < 0.35 m; split into `term/fall` and `term/self_collision` metrics | Only feet collide with the floor, so a kneeling robot sinks through the ground while still reading "upright". Termination-cause breakdown for the dashboard. |
| Metrics | `reward/<term>` sum per episode | `reward/<term>_per_step` (Brax normalises by length), `track/lin_vel_err`, `track/ang_vel_err`, `term/*`, `gait/swing_peak` | Per-term and tracking-error logging. |
| Push perturbation | enabled, 0.1–2.0 m/s every 5–10 s | present but `enable=False` for Stage 1 | Stage 1: no randomization beyond observation noise. |
| Reset randomization | xy ±0.5 m, yaw uniform, joint angles × U(0.5, 1.5), base velocity ±0.5 | same shape, ranges in `reset_config`: joint angles × U(0.8, 1.2), base velocity ±0.2 (env 1.0.2) | Upstream's ranges start most episodes half-fallen; at 30x fewer samples the policy never got past the first second. Widened again when robustness stages need it. |
| Command sampling | per-axis draws, 10 % zero, resample every 500 steps | same, ranges in config | unchanged |
| Config | `restricted_joint_range` option | removed | Only relevant with arms. |
| PPO loss | Brax `compute_ppo_loss` | fork in `muybridge/ppo_losses.py` adding `clip_fraction`, `explained_variance`, `entropy` | Dashboard PPO diagnostics. |
| Env versioning | none | `ENV_VERSION` in `g1_joystick.py`, saved to `run_config.json` and eval outputs | Comparability across checkpoints. |

## Open questions carried forward

- Upstream G1 relies on a gait phase clock (obs + `feet_phase` reward) for
  gait shaping. The spec's observation list omits it, so it is left out. If
  Stage 1 fails to produce a clean gait with `feet_air_time` alone, adding a
  phase clock is the first thing to try, and it is a one-line change to the
  interface contract that would need sign-off.
