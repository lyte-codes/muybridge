import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from muybridge import eval as eval_harness
from muybridge.envs import g1_joystick, make
from muybridge.model import load_g1


@pytest.fixture(scope="module")
def env():
  return make("G1JoystickFlat")


def test_arm_fusion_preserves_mass_and_geometry():
  full, fused = load_g1(fuse_arms=False), load_g1(fuse_arms=True)
  mf, m = full.mj_model, fused.mj_model
  assert m.nu == 15 and m.nq == 22 and m.nv == 21
  assert np.isclose(mf.body_subtreemass[1], m.body_subtreemass[1])
  df, d = mujoco.MjData(mf), mujoco.MjData(m)
  df.qpos[:] = mf.key("knees_bent").qpos
  d.qpos[:] = m.key("knees_bent").qpos
  mujoco.mj_forward(mf, df)
  mujoco.mj_forward(m, d)
  for site in ("left_palm", "right_palm"):
    np.testing.assert_allclose(df.site_xpos[mf.site(site).id], d.site_xpos[m.site(site).id], atol=1e-5)
  np.testing.assert_allclose(df.subtree_com[0], d.subtree_com[0], atol=1e-5)


def test_observation_and_action_sizes(env):
  sizes = env.observation_size
  assert env.action_size == 15
  assert sizes["state"] == (env._config.history_len * g1_joystick.FRAME_SIZE,)
  assert sizes["privileged_state"][0] > sizes["state"][0]


def test_reset_step_and_history(env):
  state = jax.jit(env.reset)(jax.random.PRNGKey(0))
  assert float(state.done) == 0.0
  step = jax.jit(env.step)
  frames = []
  for i in range(3):
    state = step(state, 0.1 * jax.random.normal(jax.random.PRNGKey(i), (15,)))
    frames.append(np.asarray(state.info["obs_history"][0]))
  hist = np.asarray(state.info["obs_history"])
  np.testing.assert_allclose(hist[0], frames[2])
  np.testing.assert_allclose(hist[1], frames[1])
  np.testing.assert_allclose(hist[2], frames[0])
  np.testing.assert_allclose(np.asarray(state.obs["state"])[: g1_joystick.FRAME_SIZE], frames[2])
  for k in env._config.reward_config.scales:
    assert f"reward/{k}_per_step" in state.metrics
  assert np.isfinite(float(state.reward))


def test_policy_obs_has_no_arm_or_linvel_terms(env):
  assert g1_joystick.FRAME_SIZE == 3 + 3 + 3 + 3 * 15
  assert g1_joystick.frame_size(env._config) == g1_joystick.FRAME_SIZE


def test_gait_phase_option_adds_clock_to_obs():
  env_phase = make("G1JoystickFlat", None, {"gait_phase.enable": True})
  assert env_phase.observation_size["state"] == (env_phase._config.history_len * (g1_joystick.FRAME_SIZE + 4),)
  state = jax.jit(env_phase.reset)(jax.random.PRNGKey(0))
  state = jax.jit(env_phase.step)(state, jp.zeros(15))
  assert np.isfinite(float(state.metrics["reward/feet_phase_per_step"]))


def test_eval_summary_pass_logic():
  T, n = 50, 120
  dt = 0.02
  recs = {
      "alive": np.ones((T, n)), "done": np.zeros((T, n)), "fall": np.zeros((T, n)), "self_collision": np.zeros((T, n)),
      "lin_err": np.full((T, n), 0.05), "ang_err": np.full((T, n), 0.05), "power": np.ones((T, n)),
      "action_delta": np.full((T, n), 0.01), "xy": np.cumsum(np.ones((T, n, 2)) * 0.01, axis=0), "torque_abs": np.ones((T, n)), "mass": 30.0,
  }
  cmds = np.tile(np.array([[0.5, 0.0, 0.0]], np.float32), (n, 1))
  seeds = np.arange(n)
  eval_harness.SETTLE_TIME = 0.2
  out = eval_harness.summarize(recs, cmds, seeds, dt, T)
  s = out["summary"]
  assert s["completion_rate"] == 1.0 and s["pass_stage1"]
  # `alive` is the pre-step flag, so the terminating step itself is still alive.
  recs["alive"][11:, 0] = 0.0
  recs["fall"][10, 0] = 1.0
  out = eval_harness.summarize(recs, cmds, seeds, dt, T)
  s = out["summary"]
  assert s["falls"] == 1 and not s["pass_stage1"]
  assert s["mean_time_to_failure_s"] == pytest.approx(11 * dt)
