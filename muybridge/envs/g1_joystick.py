"""Velocity-tracking joystick task for the Unitree G1 with fused arms.

Forked from ``mujoco_playground._src.locomotion.g1.joystick`` (Playground
0.2.0). See ``docs/upstream_diff.md`` for the list of deliberate differences.

Interface contract (SPEC.md):
  * policy input  = proprioception (stacked history) + command (vx, vy, wz)
  * policy output = position targets for 12 leg + 3 waist joints
  * arms are never observed nor actuated
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

from muybridge import model as g1_model

ENV_VERSION = "1.0.2"

NUM_JOINTS = len(g1_model.POLICY_JOINT_NAMES)  # 15
# gyro(3) + gravity(3) + command(3) + joint_pos(15) + joint_vel(15) + last_act(15)
FRAME_SIZE = 3 + 3 + 3 + 3 * NUM_JOINTS

FEET_SITES = ("left_foot", "right_foot")
FEET_GEOMS = ("left_foot", "right_foot")
TORSO_BODY = "torso_link"


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      env_version=ENV_VERSION,
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=10,
      soft_joint_pos_limit_factor=0.95,
      min_base_height=0.35,
      impl="jax",
      naconmax=None,
      njmax=None,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.75,
              lin_vel_z=-0.5,
              ang_vel_xy=-0.15,
              orientation=-2.0,
              torques=-1e-4,
              action_rate=-0.01,
              dof_acc=-1e-7,
              dof_pos_limits=-1.0,
              collision=-0.1,
              feet_air_time=2.0,
              feet_slip=-0.25,
              termination=-100.0,
              alive=1.0,
              stand_still=-1.0,
              waist_deviation=-0.2,
              joint_deviation_hip=-0.25,
              joint_deviation_knee=-0.1,
              pose=-0.1,
          ),
          tracking_sigma=0.25,
      ),
      reset_config=config_dict.create(
          xy_range=0.5,
          joint_scale_range=[0.8, 1.2],
          base_vel_range=0.2,
      ),
      push_config=config_dict.create(
          enable=False,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      command_config=config_dict.create(
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.5, 0.5],
          ang_vel_yaw=[-1.0, 1.0],
          zero_prob=0.1,
          resample_steps=500,
      ),
  )


def make_frame(
    gyro: jax.Array,
    gravity: jax.Array,
    command: jax.Array,
    joint_pos_delta: jax.Array,
    joint_vel: jax.Array,
    last_act: jax.Array,
) -> jax.Array:
  """Single-timestep proprioceptive frame; shared by training and the CPU viewer."""
  return jp.concatenate([gyro, gravity, command, joint_pos_delta, joint_vel, last_act])


def push_history(history: jax.Array, frame: jax.Array) -> jax.Array:
  """History is (H, FRAME_SIZE) with the newest frame at index 0."""
  return jp.concatenate([frame[None], history[:-1]], axis=0)


class G1Joystick(mjx_env.MjxEnv):
  """Track a (vx, vy, wz) velocity command with the leg and waist joints."""

  def __init__(
      self,
      terrain: str = "flat",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._g1 = g1_model.load_g1(terrain=terrain, fuse_arms=True, sim_dt=self.sim_dt)
    self._mj_model = self._g1.mj_model
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = self._g1.xml_path
    self._terrain = terrain
    self._post_init()

  def _post_init(self) -> None:
    m = self._mj_model
    self._init_q = jp.array(m.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(m.keyframe("knees_bent").qpos[7:])
    assert self._default_pose.shape == (NUM_JOINTS,)

    self._lowers, self._uppers = m.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    def qidx(names):
      return jp.array([m.joint(n).qposadr[0] - 7 for n in names])

    self._waist_indices = qidx(g1_model.WAIST_JOINT_NAMES)
    self._hip_indices = qidx([f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_roll", "hip_yaw")])
    self._knee_indices = qidx([f"{s}_knee_joint" for s in ("left", "right")])

    self._torso_body_id = m.body(TORSO_BODY).id
    self._torso_imu_site_id = m.site("imu_in_torso").id
    self._pelvis_imu_site_id = m.site("imu_in_pelvis").id
    self._feet_site_id = np.array([m.site(n).id for n in FEET_SITES])
    self._floor_geom_id = m.geom("floor").id
    self._feet_geom_id = np.array([m.geom(n).id for n in FEET_GEOMS])

    foot_linvel_sensor_adr = []
    for site in FEET_SITES:
      sid = m.sensor(f"{site}_global_linvel").id
      adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
      foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    def sensor_adr(name):
      return int(m.sensor_adr[m.sensor(name).id])

    self._feet_floor_found_adr = jp.array([sensor_adr(f"{g}_floor_found") for g in FEET_GEOMS])
    self._self_collision_adr = jp.array([
        sensor_adr("right_foot_left_foot_found"),
        sensor_adr("left_foot_right_shin_found"),
        sensor_adr("right_foot_left_shin_found"),
    ])
    self._hand_thigh_adr = jp.array([
        sensor_adr("left_hand_left_thigh_found"),
        sensor_adr("right_hand_right_thigh_found"),
    ])
    self._cmd_ranges = jp.array([
        self._config.command_config.lin_vel_x,
        self._config.command_config.lin_vel_y,
        self._config.command_config.ang_vel_yaw,
    ])

  # Sensors.

  def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, name)

  def get_gravity(self, data, frame="torso"):
    return self._sensor(data, f"upvector_{frame}")

  def get_global_linvel(self, data, frame="pelvis"):
    return self._sensor(data, f"global_linvel_{frame}")

  def get_global_angvel(self, data, frame="pelvis"):
    return self._sensor(data, f"global_angvel_{frame}")

  def get_local_linvel(self, data, frame="pelvis"):
    return self._sensor(data, f"local_linvel_{frame}")

  def get_accelerometer(self, data, frame="pelvis"):
    return self._sensor(data, f"accelerometer_{frame}")

  def get_gyro(self, data, frame="pelvis"):
    return self._sensor(data, f"gyro_{frame}")

  def feet_contact(self, data: mjx.Data) -> jax.Array:
    return data.sensordata[self._feet_floor_found_adr] > 0

  # Episode lifecycle.

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    rc = self._config.reset_config
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-rc.xy_range, maxval=rc.xy_range)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (NUM_JOINTS,), minval=rc.joint_scale_range[0], maxval=rc.joint_scale_range[1])
    )
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-rc.base_vel_range, maxval=rc.base_vel_range))

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(NUM_JOINTS),
        "last_last_act": jp.zeros(NUM_JOINTS),
        "motor_targets": self._default_pose,
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "obs_history": jp.zeros((self._config.history_len, FRAME_SIZE)),
        "push": jp.zeros(2),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}_per_step"] = jp.zeros(())
    metrics["track/lin_vel_err_per_step"] = jp.zeros(())
    metrics["track/ang_vel_err_per_step"] = jp.zeros(())
    metrics["term/fall"] = jp.zeros(())
    metrics["term/self_collision"] = jp.zeros(())
    metrics["gait/swing_peak_per_step"] = jp.zeros(())

    contact = self.feet_contact(data)
    frame = self._frame(data, info, noisy=True)
    info["obs_history"] = jp.tile(frame[None], (self._config.history_len, 1))
    obs = self._get_obs(data, info, contact, frame)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    state = state.replace(data=state.data.replace(qvel=qvel))

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
    state.info["motor_targets"] = motor_targets

    contact = self.feet_contact(data)
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_fz = data.site_xpos[self._feet_site_id][..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    frame = self._frame(data, state.info, noisy=True)
    state.info["obs_history"] = push_history(state.info["obs_history"], frame)
    obs = self._get_obs(data, state.info, contact, frame)

    fall, self_collision = self._termination_causes(data)
    done = fall | self_collision

    rewards = self._get_reward(data, action, state.info, done, first_contact, contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = sum(rewards.values()) * self.dt

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    resample = state.info["step"] > self._config.command_config.resample_steps
    state.info["command"] = jp.where(resample, self.sample_command(cmd_rng), state.info["command"])
    state.info["step"] = jp.where(done | resample, 0, state.info["step"])
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact

    for k, v in rewards.items():
      state.metrics[f"reward/{k}_per_step"] = v
    local_linvel = self.get_local_linvel(data)
    state.metrics["track/lin_vel_err_per_step"] = jp.linalg.norm(state.info["command"][:2] - local_linvel[:2])
    state.metrics["track/ang_vel_err_per_step"] = jp.abs(state.info["command"][2] - self.get_gyro(data)[2])
    state.metrics["term/fall"] = fall.astype(jp.float32)
    state.metrics["term/self_collision"] = self_collision.astype(jp.float32)
    state.metrics["gait/swing_peak_per_step"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _termination_causes(self, data: mjx.Data):
    fall = self.get_gravity(data, "torso")[-1] < 0.0
    # Only the feet collide with the floor, so a robot that drops onto its
    # knees sinks through the ground while its torso can still read "upright".
    fall |= data.qpos[2] < self._config.min_base_height
    fall |= jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    self_collision = jp.any(data.sensordata[self._self_collision_adr] > 0)
    return fall, self_collision

  # Observations.

  def _noisy(self, info: dict, x: jax.Array, scale: float) -> jax.Array:
    info["rng"], key = jax.random.split(info["rng"])
    noise = (2 * jax.random.uniform(key, shape=x.shape) - 1) * self._config.noise_config.level * scale
    return x + noise

  def _frame(self, data: mjx.Data, info: dict, noisy: bool) -> jax.Array:
    scales = self._config.noise_config.scales
    gyro = self.get_gyro(data)
    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1.0])
    joint_pos = data.qpos[7:] - self._default_pose
    joint_vel = data.qvel[6:]
    if noisy:
      gyro = self._noisy(info, gyro, scales.gyro)
      gravity = self._noisy(info, gravity, scales.gravity)
      joint_pos = self._noisy(info, joint_pos, scales.joint_pos)
      joint_vel = self._noisy(info, joint_vel, scales.joint_vel)
    return make_frame(gyro, gravity, info["command"], joint_pos, joint_vel, info["last_act"])

  def _get_obs(self, data: mjx.Data, info: dict, contact: jax.Array, noisy_frame: jax.Array) -> Dict[str, jax.Array]:
    del noisy_frame
    state = info["obs_history"].ravel()
    clean_frame = self._frame(data, info, noisy=False)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    privileged_state = jp.hstack([
        state,
        clean_frame,
        self.get_local_linvel(data),
        self.get_global_angvel(data),
        self.get_accelerometer(data),
        data.qpos[2],
        data.actuator_force,
        contact,
        feet_vel,
        info["feet_air_time"],
    ])
    return {"state": state, "privileged_state": privileged_state}

  # Rewards.

  def _get_reward(self, data, action, info, done, first_contact, contact) -> Dict[str, jax.Array]:
    cmd = info["command"]
    qpos = data.qpos[7:]
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(cmd, self.get_local_linvel(data)),
        "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, self.get_gyro(data)),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data, "pelvis"), self.get_global_linvel(data, "torso")),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data, "torso")),
        "orientation": self._cost_orientation(self.get_gravity(data, "torso")),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "dof_pos_limits": self._cost_joint_pos_limits(qpos),
        "collision": self._cost_collision(data),
        "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact),
        "feet_slip": self._cost_feet_slip(data, contact),
        "termination": done,
        "alive": jp.array(1.0),
        "stand_still": self._cost_stand_still(cmd, qpos),
        "waist_deviation": self._cost_waist_deviation(qpos),
        "joint_deviation_hip": self._cost_joint_deviation_hip(qpos, cmd),
        "joint_deviation_knee": self._cost_joint_deviation_knee(qpos),
        "pose": self._cost_pose(qpos),
    }

  def _reward_tracking_lin_vel(self, cmd, local_vel):
    err = jp.sum(jp.square(cmd[:2] - local_vel[:2]))
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(self, cmd, ang_vel):
    err = jp.square(cmd[2] - ang_vel[2])
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _cost_lin_vel_z(self, linvel_pelvis, linvel_torso):
    return jp.square(linvel_pelvis[2]) + jp.square(linvel_torso[2])

  def _cost_ang_vel_xy(self, angvel_torso):
    return jp.sum(jp.square(angvel_torso[:2]))

  def _cost_orientation(self, torso_zaxis):
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_torques(self, torques):
    return jp.sum(jp.abs(torques))

  def _cost_action_rate(self, act, last_act):
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc):
    return jp.sum(jp.square(qacc))

  def _cost_joint_pos_limits(self, qpos):
    out = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out)

  def _cost_collision(self, data):
    return jp.any(data.sensordata[self._hand_thigh_adr] > 0).astype(jp.float32)

  def _reward_feet_air_time(self, air_time, first_contact, threshold_min=0.2, threshold_max=0.5):
    air_time = (air_time - threshold_min) * first_contact
    return jp.sum(jp.clip(air_time, max=threshold_max - threshold_min))

  def _cost_feet_slip(self, data, contact):
    body_vel = self.get_global_linvel(data, "pelvis")[:2]
    return jp.sum(jp.linalg.norm(body_vel) * contact)

  def _cost_stand_still(self, cmd, qpos):
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    return cost * (jp.linalg.norm(cmd) < 0.01)

  def _cost_waist_deviation(self, qpos):
    return jp.sum(jp.abs(qpos[self._waist_indices] - self._default_pose[self._waist_indices]))

  def _cost_joint_deviation_hip(self, qpos, cmd):
    error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    weight = jp.where(jp.abs(cmd[1]) > 0.1, jp.array([0.0, 1.0, 0.0, 1.0]), jp.ones(4))
    return jp.sum(jp.abs(error) * weight)

  def _cost_joint_deviation_knee(self, qpos):
    return jp.sum(jp.abs(qpos[self._knee_indices] - self._default_pose[self._knee_indices]))

  def _cost_pose(self, qpos):
    return jp.sum(jp.square(qpos - self._default_pose))

  # Commands.

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2 = jax.random.split(rng)
    cmd = jax.random.uniform(rng1, (3,), minval=self._cmd_ranges[:, 0], maxval=self._cmd_ranges[:, 1])
    zero = jax.random.bernoulli(rng2, p=self._config.command_config.zero_prob)
    return jp.where(zero, jp.zeros(3), cmd)

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def default_pose(self) -> jax.Array:
    return self._default_pose

  @property
  def terrain(self) -> str:
    return self._terrain
