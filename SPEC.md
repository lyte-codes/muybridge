# Handoff: Morphology-Robust Humanoid Locomotion in MJX

## Goal

Train a velocity-tracking locomotion policy for a commodity humanoid that is
robust to (a) terrain variation, (b) morphology variation around a nominal
humanoid, and (c) unmodeled upper-body disturbance.

**Sim only.** No hardware deployment, no sim-to-real validation. Success is
measured entirely in simulation under held-out randomization.

## Non-goals — do not implement these

- Arm or hand control of any kind
- Manipulation, grasping, object interaction
- Vision, cameras, rendering, depth, or any image observation
- Whole-body control coupling legs and arms in a single policy
- Real robot deployment, ROS, hardware drivers
- Custom robot design or URDF authoring

If a task seems to require any of the above, stop and ask rather than
improvising.

## Stack

- **Physics:** MuJoCo + MJX (JAX backend). Must run on Apple silicon.
- **Robot:** Unitree G1 from MuJoCo Menagerie. Do not modify the base URDF;
  apply variation at runtime via randomization.
- **Algorithm:** PPO.
- **Starting point:** Fork the locomotion examples in MuJoCo Playground rather
  than writing environments from scratch. Diff against upstream so we can see
  what actually changed.

### Hardware constraints

Development machine is an M2 MacBook Air, 16 GB unified memory, no CUDA.

- Isaac Lab, Isaac Sim, and anything PhysX-based are unavailable. Do not
  suggest them.
- Keep parallel env count configurable; assume small batches (64–512) locally.
- Long runs may be pushed to a rented NVIDIA GPU. Code must not hardcode a
  backend or assume Metal.
- JIT compile time on MJX is significant. Cache compiled functions and do not
  re-JIT per episode.

## Interface contract — hold this stable across all stages

The locomotion policy is one component of a larger system. Fix these
boundaries now so the arm controller can be swapped later without retraining.

- **Policy input:** proprioception + a 3-vector velocity command
  `(vx, vy, omega_z)`.
- **Policy output:** joint position targets for leg **and waist** joints,
  tracked by a PD controller. Not torques. Waist joints are part of the
  locomotion policy and give it authority to counteract torso disturbance.
  Include a small waist-deviation penalty in the reward so the policy does not
  use torso motion as a balance crutch on flat ground.
- **Upper body:** treated as an external disturbance on the torso, never as an
  observation or an action. The policy must not receive arm state.

## Stages

Each stage must pass its criteria before the next begins. Commit at each
boundary.

### Stage 0 — Environment

Get MJX running locally with the G1 model loaded. Verify a random-action
rollout completes, measure steps/sec at 64/256/512 envs, and report it. No
training.

**Pass:** rollout runs, throughput numbers reported.

### Stage 1 — Flat-ground velocity tracking

PPO on flat ground, nominal G1, no randomization beyond observation noise.

Observations: base angular velocity, projected gravity, velocity command,
joint positions, joint velocities, previous action.

**Use a stacked observation history from the start** (e.g. last 10–15
timesteps), even though Stage 1 does not strictly need it. Stage 3 requires
the policy to infer its own body properties from dynamics, which is impossible
from a single timestep. Retrofitting history later invalidates Stages 1–2.

Reward terms (tune weights, keep the set):
- `tracking_lin_vel` — exp kernel on `(vx, vy)` command error
- `tracking_ang_vel` — exp kernel on `omega_z` command error
- penalties: vertical linear velocity, roll/pitch angular velocity,
  non-flat base orientation, torque, action rate, joint acceleration,
  joint limit violation, self-collision
- `feet_air_time` reward to discourage foot scuffing
- termination penalty on fall

**Pass:** tracks commanded velocity within 0.1 m/s over a 20 s episode across
a command grid spanning ±1.0 m/s forward, ±0.5 m/s lateral, ±1.0 rad/s yaw.
No falls in 100 consecutive evaluation episodes.

### Stage 2 — Rough terrain

Add procedural terrain: slopes, steps, and random rough. Curriculum from flat
to difficult based on tracking performance.

Optionally add a local height scan to observations. If added, keep it as a
separate, ablatable observation group — we need to know whether the policy
depends on it.

**Pass:** ≥90% episode completion on held-out terrain seeds not used in
training.

### Stage 3 — Morphology randomization

This is the "commodity humanoid" requirement. Randomize per-episode around the
G1 nominal:

- link masses ±25%
- link lengths ±10%
- center-of-mass offsets per link
- actuator strength / torque limits 0.8–1.2×
- PD gains ±20%
- joint damping and friction
- ground friction 0.3–1.5
- control latency 0–20 ms

**Pass:** ≥85% episode completion on a held-out set of morphology samples
drawn from the same distribution but never seen in training. Report the
performance gap between nominal and randomized — that gap is the headline
result.

### Stage 4 — Torso disturbance (arm proxy)

Model the upper body as unmodeled load. **Do not simulate an arm.** Adding arm
bodies costs physics time in every parallel env to produce a signal the policy
never observes, and invites overfitting to one arm's dynamics.

Instead, parameterize the disturbance by its statistics:

- Per-episode randomized payload mass (0–5 kg) at randomized torso CoM offset
- Continuous smooth CoM/wrench variation in the 0.5–2 Hz band, amplitude
  scaled to a plausible payload at plausible arm extension
- Occasional step changes to emulate pick and place transitions
- Random impulse pushes

Expose amplitude and bandwidth as sweepable parameters and report which
regions of that space the policy survives. That sweep is a result in itself.

**Pass:** ≥80% episode completion across the disturbance parameter sweep at
payloads up to 5 kg.

### Stage 5 — IK validation (optional, after Stage 4)

Attach a kinematically real arm to the torso and drive it with inverse
kinematics through actual reach trajectories. No learned arm policy, no
manipulation. This checks whether the Stage 4 disturbance model was
representative.

If the legs hold, the model was adequate. If they fail, the disturbance
parameterization was too tame — widen it and retrain Stage 4. Do not tune the
IK trajectories to make the legs succeed.

## Evaluation

Build the eval harness in Stage 1, not later. It must be fixed and reusable
across all stages:

- Fixed seed set, held out from training
- Fixed command grid
- Metrics: velocity tracking error (linear, angular), episode completion rate,
  mean time to failure, cost of transport, action smoothness
- Results written to a versioned file per run, comparable across checkpoints

A checkpoint is better than its predecessor only by this harness. Do not
compare training-time reward curves across configs.

## Dashboard

Build this alongside Stage 1, not after.

**Architectural constraint:** MJX training runs as a compiled JAX loop over
many parallel envs. Do not render during training — it will destroy
throughput and force recompilation. The dashboard is two decoupled processes.

### Process 1 — metrics (live)

Training writes scalars to disk continuously. Surface at minimum:

- Each reward term logged *separately*, not just total reward. Without this,
  reward tuning is guesswork — most failures show up as one term dominating.
- Episode length, termination cause breakdown (fall, timeout, joint limit)
- Velocity tracking error, linear and angular, separately
- PPO diagnostics: policy loss, value loss, entropy, approximate KL, clip
  fraction, explained variance
- Curriculum level (Stage 2+), randomization ranges in effect
- Throughput: env steps/sec, wall-clock per iteration

TensorBoard or Weights & Biases is sufficient here. Do not build a custom
metrics UI.

### Process 2 — 3D rollout viewer (required)

This is a required deliverable, not optional tooling. I need to watch the
robot move and see what the policy is doing, not just read scalars.

Separate process from training. Loads the most recent checkpoint, runs a
*single* environment in standard MuJoCo on CPU, and renders it in 3D. Polls
the checkpoint directory and hot-reloads when a newer one appears.

**Use Viser** for a browser-based 3D view — orbit, pan, zoom, works over the
network, no local GUI dependency. Fall back to `mujoco.viewer` only if Viser
proves impractical.

Requirements:

- Free camera control, plus a toggle to lock the camera to follow the torso
- Play, pause, step, and reset controls
- Manual velocity command input — sliders or keyboard — so I can drive the
  policy around and probe it directly rather than watching scripted commands
- A checkpoint selector, so I can load an older checkpoint and compare
  behaviour against the current one
- Terrain rendered as actually simulated (Stage 2+), not a flat placeholder

Overlays on the 3D view:

- Commanded velocity vector and actual velocity vector, drawn as arrows from
  the torso
- Per-foot contact force, as a scalar bar or a scaled arrow at each foot
- Foot contact state and swing/stance phase
- Center of mass and support polygon projected on the ground plane
- The active randomization sample for the episode — morphology parameters,
  payload mass, CoM offset — displayed as text
- Applied torso disturbance wrench (Stage 4+), drawn as an arrow

The CoM-versus-support-polygon overlay is the one that will tell you most
about why a policy falls. Prioritise it.

This is not real-time in the sense of showing the actual training envs. It
shows what the current policy does, refreshed every checkpoint. That
distinction is fine and is the only approach that doesn't slow training.

### Checkpoint cadence

Frequent enough that the viewer feels responsive, infrequent enough not to
stall training. Start at every 50 iterations and tune.

## Reporting

At each stage boundary, write a short summary: what changed, eval numbers
versus the previous stage, what regressed, and open questions. Keep it in the
repo.

## Open questions to raise rather than guess

- Whether the height scan in Stage 2 is worth the observation-space cost
- Whether morphology randomization ranges in Stage 3 are too wide to learn
  (if Stage 3 fails to converge, narrow and report the boundary rather than
  abandoning)
- Whether PPO is sufficient or a different algorithm is warranted — raise
  with evidence, do not switch unilaterally

## Git workflow

- `main` is the primary branch and holds only work that has passed a stage's
  criteria. Never commit to `main` directly.
- One branch per stage, named `stage-N-short-description`, branched from
  `main`. Open a draft PR immediately and push incrementally.
- Fix all issues within the stage's own branch; never defer them to the next
  stage. Mark the PR ready only when the eval harness confirms that stage's
  pass criteria. Then squash merge, delete the branch, and tag (`v0.1` for
  stage 1, `v0.2` for stage 2, and so on).
- PR descriptions include: what changed, eval numbers vs. the previous stage,
  anything that regressed, whether the env version was bumped and why, and
  open questions.
