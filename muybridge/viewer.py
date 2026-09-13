"""Browser-based 3D rollout viewer (Viser). Runs one env in CPU MuJoCo.

    python -m muybridge.viewer --run runs/stage1_a --port 8080

Loads the newest checkpoint in ``<run>/checkpoints`` (hot-reloads when a newer
one appears, or pick any checkpoint from the dropdown), drives the policy with
manual velocity sliders, and overlays command/actual velocity, foot contact
forces and phase, CoM vs. support polygon, the active randomization sample,
and any applied torso wrench.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import threading
import time
from typing import Dict, List, Optional

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import viser

import muybridge  # noqa: F401
from muybridge import checkpoints, envs
from muybridge.envs import g1_joystick

NUM_JOINTS = g1_joystick.NUM_JOINTS


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  out = np.zeros(9)
  mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
  return out.reshape(3, 3)


# ----------------------------------------------------------------------------- simulation


class Sim:
  """Single CPU MuJoCo env mirroring G1Joystick's observation and control pipeline."""

  def __init__(self, env, noise_level: float):
    self.env = env
    self.m = env.mj_model
    self.d = mujoco.MjData(self.m)
    self.dt = env.dt
    self.n_substeps = env.n_substeps
    self.action_scale = float(env._config.action_scale)
    self.default_pose = np.asarray(env.default_pose)
    self.history_len = int(env._config.history_len)
    self.noise_scales = env._config.noise_config.scales
    self.noise_level = noise_level
    self.rng = np.random.default_rng(0)
    self.phase_enabled = bool(env._config.gait_phase.enable)
    self.phase_dt = 2 * np.pi * self.dt * float(np.mean(env._config.gait_phase.freq_range))
    self.pelvis_imu = self.m.site("imu_in_pelvis").id
    self.torso_body = self.m.body(g1_joystick.TORSO_BODY).id
    self.foot_geoms = [self.m.geom(n).id for n in g1_joystick.FEET_GEOMS]
    self.foot_sites = [self.m.site(n).id for n in g1_joystick.FEET_SITES]
    self.floor_geom = self.m.geom("floor").id
    self.total_mass = float(self.m.body_subtreemass[1])
    self.randomization_sample: Dict[str, object] = {"sample": "nominal (no randomization in this stage)"}
    self.reset()

  def sensor(self, name: str) -> np.ndarray:
    sid = self.m.sensor(name).id
    adr, dim = self.m.sensor_adr[sid], self.m.sensor_dim[sid]
    return self.d.sensordata[adr:adr + dim].copy()

  def reset(self, yaw: float = 0.0) -> None:
    mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("knees_bent").id)
    self.d.qvel[:] = 0
    self.d.ctrl[:] = self.default_pose
    mujoco.mj_forward(self.m, self.d)
    self.command = np.zeros(3)
    self.last_act = np.zeros(NUM_JOINTS)
    self.step_count = 0
    self.air_time = np.zeros(2)
    self.fallen = False
    self.phase = np.array([0.0, np.pi])
    self.history = np.tile(self.frame()[None], (self.history_len, 1))

  def noisy(self, x: np.ndarray, scale: float) -> np.ndarray:
    return x + (2 * self.rng.uniform(size=x.shape) - 1) * self.noise_level * scale

  def frame(self) -> np.ndarray:
    gyro = self.noisy(self.sensor("gyro_pelvis"), self.noise_scales.gyro)
    gravity = self.noisy(self.d.site_xmat[self.pelvis_imu].reshape(3, 3).T @ np.array([0, 0, -1.0]), self.noise_scales.gravity)
    jpos = self.noisy(self.d.qpos[7:] - self.default_pose, self.noise_scales.joint_pos)
    jvel = self.noisy(self.d.qvel[6:].copy(), self.noise_scales.joint_vel)
    phase = self.phase if self.phase_enabled else None
    return np.asarray(g1_joystick.make_frame(gyro, gravity, self.command, jpos, jvel, self.last_act, phase))

  def obs(self, obs_sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    out = {"state": self.history.ravel().astype(np.float32)}
    for k, n in obs_sizes.items():
      if k != "state":
        out[k] = np.zeros(n, np.float32)
    return out

  def step(self, action: np.ndarray) -> None:
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    self.d.ctrl[:] = self.default_pose + action * self.action_scale
    for _ in range(self.n_substeps):
      mujoco.mj_step(self.m, self.d)
    self.last_act = action
    self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
    self.history = np.asarray(g1_joystick.push_history(self.history, self.frame()))
    self.step_count += 1
    contact = self.foot_contacts()
    self.air_time = np.where(contact, 0.0, self.air_time + self.dt)
    self.fallen = bool(self.sensor("upvector_torso")[2] < 0.0) or bool(np.isnan(self.d.qpos).any())

  # Readouts.

  def foot_contacts(self) -> np.ndarray:
    return np.array([self.sensor(f"{g}_floor_found")[0] > 0 for g in g1_joystick.FEET_GEOMS])

  def foot_contact_forces(self):
    """World-frame ground reaction force per foot and the list of contact points."""
    forces = np.zeros((2, 3))
    points: List[np.ndarray] = []
    f6 = np.zeros(6)
    for i in range(self.d.ncon):
      c = self.d.contact[i]
      for k, fg in enumerate(self.foot_geoms):
        if fg in (c.geom1, c.geom2):
          mujoco.mj_contactForce(self.m, self.d, i, f6)
          frame = np.asarray(c.frame).reshape(3, 3)
          f_world = frame.T @ f6[:3]
          if c.geom2 == fg:
            f_world = -f_world
          forces[k] += f_world
          points.append(np.array(c.pos))
    return forces, points

  def local_linvel(self) -> np.ndarray:
    return self.sensor("local_linvel_pelvis")

  def gyro(self) -> np.ndarray:
    return self.sensor("gyro_pelvis")

  def com(self) -> np.ndarray:
    return self.d.subtree_com[0].copy()

  def torso_pos(self) -> np.ndarray:
    return self.d.xpos[self.torso_body].copy()

  def applied_wrench(self) -> np.ndarray:
    return self.d.xfrc_applied[self.torso_body].copy()

  def ground_height(self, x: float, y: float) -> float:
    fid = self.floor_geom
    if self.m.geom_type[fid] != mujoco.mjtGeom.mjGEOM_HFIELD:
      return float(self.m.geom_pos[fid][2])
    hid = self.m.geom_dataid[fid]
    nrow, ncol = self.m.hfield_nrow[hid], self.m.hfield_ncol[hid]
    rx, ry, zmax, _ = self.m.hfield_size[hid]
    gx, gy = self.m.geom_pos[fid][:2]
    i = int(np.clip((y - gy + ry) / (2 * ry) * (nrow - 1), 0, nrow - 1))
    j = int(np.clip((x - gx + rx) / (2 * rx) * (ncol - 1), 0, ncol - 1))
    return float(self.m.hfield_data[hid * nrow * ncol + i * ncol + j] * zmax + self.m.geom_pos[fid][2])


# ----------------------------------------------------------------------------- policy


class Policy:

  def __init__(self, path: str):
    self.path = path
    self.step = checkpoints.checkpoint_step(path)
    self.obs_sizes = checkpoints.observation_sizes(path)
    fn = checkpoints.load_policy(path, deterministic=True)
    self._fn = jax.jit(lambda obs, key: fn(obs, key)[0])
    self._key = jax.random.PRNGKey(0)
    self(self._zeros())

  def _zeros(self):
    return {k: np.zeros(n, np.float32) for k, n in self.obs_sizes.items()}

  def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(self._fn({k: jp.asarray(v) for k, v in obs.items()}, self._key))


# ----------------------------------------------------------------------------- geometry


def body_meshes(m: mujoco.MjModel, groups=(0, 1, 2)) -> Dict[int, trimesh.Trimesh]:
  parts: Dict[int, List[trimesh.Trimesh]] = {}
  for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or int(m.geom_group[g]) not in groups:
      continue
    mid = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    verts = m.mesh_vert[va:va + vn] @ quat_to_mat(m.geom_quat[g]).T + m.geom_pos[g]
    faces = m.mesh_face[fa:fa + fn]
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.visual.face_colors = np.tile((np.asarray(rgba) * 255).astype(np.uint8), (len(faces), 1))
    parts.setdefault(int(m.geom_bodyid[g]), []).append(tm)
  return {b: trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0] for b, ms in parts.items()}


def collision_meshes(m: mujoco.MjModel) -> Dict[int, trimesh.Trimesh]:
  parts: Dict[int, List[trimesh.Trimesh]] = {}
  for g in range(m.ngeom):
    if int(m.geom_group[g]) != 3 or m.geom_bodyid[g] == 0:
      continue
    t, size = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
      tm = trimesh.creation.box(extents=2 * size)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      tm = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
      tm = trimesh.creation.icosphere(radius=size[0])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
      tm = trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    else:
      continue
    tm.apply_transform(trimesh.transformations.concatenate_matrices(
        trimesh.transformations.translation_matrix(m.geom_pos[g]),
        np.block([[quat_to_mat(m.geom_quat[g]), np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]),
    ))
    tm.visual.face_colors = np.tile(np.array([60, 200, 60, 90], np.uint8), (len(tm.faces), 1))
    parts.setdefault(int(m.geom_bodyid[g]), []).append(tm)
  return {b: trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0] for b, ms in parts.items()}


def hfield_mesh(m: mujoco.MjModel, geom_id: int, max_dim: int = 256):
  hid = m.geom_dataid[geom_id]
  nrow, ncol = m.hfield_nrow[hid], m.hfield_ncol[hid]
  rx, ry, zmax, _ = m.hfield_size[hid]
  data = m.hfield_data[hid * nrow * ncol:(hid + 1) * nrow * ncol].reshape(nrow, ncol)
  stride = max(1, int(np.ceil(max(nrow, ncol) / max_dim)))
  data = data[::stride, ::stride]
  r, c = data.shape
  xs = np.linspace(-rx, rx, c) + m.geom_pos[geom_id][0]
  ys = np.linspace(-ry, ry, r) + m.geom_pos[geom_id][1]
  xx, yy = np.meshgrid(xs, ys)
  verts = np.stack([xx, yy, data * zmax + m.geom_pos[geom_id][2]], axis=-1).reshape(-1, 3)
  idx = np.arange(r * c).reshape(r, c)
  a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
  faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, cc], -1).reshape(-1, 3)])
  return verts.astype(np.float32), faces.astype(np.int32)


def support_polygon(points: List[np.ndarray]) -> Optional[np.ndarray]:
  if not points:
    return None
  xy = np.array([p[:2] for p in points])
  if len(xy) < 3:
    return xy
  try:
    hull = ConvexHull(xy)
  except Exception:  # noqa: BLE001 (degenerate/collinear)
    return xy
  return xy[hull.vertices]


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
  if poly is None or len(poly) < 3:
    return False
  inside = False
  n = len(poly)
  for i in range(n):
    a, b = poly[i], poly[(i + 1) % n]
    if (a[1] > p[1]) != (b[1] > p[1]):
      x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-12)
      if p[0] < x:
        inside = not inside
  return inside


# ----------------------------------------------------------------------------- viewer


class Viewer:

  def __init__(self, run_dir: str, env_name: str, port: int, noise_level: float):
    self.run_dir = run_dir
    self.ckpt_dir = os.path.join(run_dir, "checkpoints")
    self.env = envs.make(env_name)
    self.sim = Sim(self.env, noise_level)
    self.policy: Optional[Policy] = None
    self.policy_lock = threading.Lock()
    self.server = viser.ViserServer(port=port, label="muybridge viewer")
    self.server.scene.set_up_direction("+z")
    self.server.initial_camera.position = tuple(self.sim.torso_pos() + np.array([2.2, -2.2, 1.0]))
    self.server.initial_camera.look_at = tuple(self.sim.torso_pos())
    self.playing = True
    self.step_request = False
    self.reset_request = False
    self.fall_time: Optional[float] = None
    self.client_offsets: Dict[int, np.ndarray] = {}
    self.commanded_positions: Dict[int, collections.deque] = {}
    self.commanded_targets: Dict[int, collections.deque] = {}
    self.camera_offset0 = np.array([2.2, -2.2, 1.0])
    self.auto_ckpt = True
    self._build_scene()
    self._build_gui()
    self._load_latest()
    self.server.on_client_connect(self._on_client_connect)
    threading.Thread(target=self._poll_checkpoints, daemon=True).start()

  def _on_client_connect(self, client: viser.ClientHandle) -> None:
    cid = client.client_id
    self.client_offsets[cid] = self.camera_offset0.copy()
    self.commanded_positions[cid] = collections.deque(maxlen=30)
    self.commanded_targets[cid] = collections.deque(maxlen=30)

    @client.camera.on_update
    def _(cam: viser.CameraHandle) -> None:
      # User orbit/zoom keeps look_at on a target we commanded but moves the
      # position away from anything we commanded; everything else is an echo.
      pos, look_at = np.asarray(cam.position), np.asarray(cam.look_at)
      if not any(np.linalg.norm(look_at - t) < 1e-3 for t in self.commanded_targets[cid]):
        return
      if any(np.linalg.norm(pos - c) < 1e-4 for c in self.commanded_positions[cid]):
        return
      self.client_offsets[cid] = pos - look_at

  # Scene.

  def _build_scene(self) -> None:
    m = self.sim.m
    self.server.scene.add_light_directional("/light", position=(3.0, 3.0, 6.0), intensity=2.0, cast_shadow=True)
    self.server.scene.add_light_ambient("/ambient", intensity=0.6)
    floor = m.geom("floor").id
    if m.geom_type[floor] == mujoco.mjtGeom.mjGEOM_HFIELD:
      verts, faces = hfield_mesh(m, floor)
      self.server.scene.add_mesh_simple("/terrain", verts, faces, color=(190, 175, 150), flat_shading=True, side="double")
    else:
      z = float(m.geom_pos[floor][2])
      self.server.scene.add_grid("/terrain", width=60.0, height=60.0, plane="xy", cell_size=0.5, section_size=2.5, position=(0, 0, z), plane_color=(235, 235, 235), plane_opacity=1.0)
    self.body_handles = {}
    for b, tm in body_meshes(m).items():
      self.body_handles[b] = self.server.scene.add_mesh_trimesh(f"/robot/{m.body(b).name}", tm)
    self.collision_handles = {}
    for b, tm in collision_meshes(m).items():
      self.collision_handles[b] = self.server.scene.add_mesh_trimesh(f"/collision/{m.body(b).name}", tm, visible=False)
    self.com_handle = self.server.scene.add_icosphere("/overlay/com", radius=0.03, color=(230, 40, 40))
    self.com_ground_handle = self.server.scene.add_icosphere("/overlay/com_ground", radius=0.025, color=(230, 40, 40))
    self.foot_handles = [self.server.scene.add_icosphere(f"/overlay/foot{i}", radius=0.02, color=(120, 120, 120)) for i in range(2)]
    self.foot_labels = [None, None]
    self.foot_phase = ["", ""]

  def _update_scene(self) -> None:
    d = self.sim.d
    with self.server.atomic():
      for b, h in self.body_handles.items():
        h.position = d.xpos[b]
        h.wxyz = d.xquat[b]
      if self.show_collision.value:
        for b, h in self.collision_handles.items():
          h.position = d.xpos[b]
          h.wxyz = d.xquat[b]
      if self.show_overlays.value:
        self._update_overlays()

  def _update_overlays(self) -> None:
    scene = self.server.scene
    sim = self.sim
    torso = sim.torso_pos()
    yaw_mat = quat_to_mat(sim.d.qpos[3:7])
    cmd = sim.command
    cmd_world = yaw_mat @ np.array([cmd[0], cmd[1], 0.0])
    vel_world = sim.sensor("global_linvel_pelvis")
    vel_world[2] = 0.0
    origin = torso + np.array([0, 0, 0.35])
    scene.add_arrows("/overlay/cmd_vel", np.array([[origin, origin + cmd_world]]), colors=(40, 90, 255), shaft_radius=0.012, head_radius=0.03, head_length=0.06)
    scene.add_arrows("/overlay/act_vel", np.array([[origin, origin + vel_world]]), colors=(40, 200, 80), shaft_radius=0.012, head_radius=0.03, head_length=0.06)
    yaw_arrow_len = 0.3
    scene.add_arrows("/overlay/cmd_yaw", np.array([[origin, origin + np.array([0, 0, cmd[2] * yaw_arrow_len])]]), colors=(140, 90, 255), shaft_radius=0.008, head_radius=0.02, head_length=0.04)

    forces, points = sim.foot_contact_forces()
    contact = sim.foot_contacts()
    foot_pos = sim.d.site_xpos[sim.foot_sites]
    scale = 1.0 / (sim.total_mass * 9.81)
    arrows = np.array([[foot_pos[i], foot_pos[i] + forces[i] * scale] for i in range(2)])
    scene.add_arrows("/overlay/grf", arrows, colors=(255, 140, 0), shaft_radius=0.01, head_radius=0.025, head_length=0.05)
    for i in range(2):
      self.foot_handles[i].position = foot_pos[i]
      phase = "stance" if contact[i] else "swing"
      if phase != self.foot_phase[i]:
        self.foot_phase[i] = phase
        self.foot_handles[i].remove()
        self.foot_handles[i] = scene.add_icosphere(f"/overlay/foot{i}", radius=0.02, color=(40, 200, 80) if contact[i] else (150, 150, 150), position=foot_pos[i])
        self.foot_labels[i] = scene.add_label(f"/overlay/foot_label{i}", f"{'L' if i == 0 else 'R'} {phase}", position=foot_pos[i] + np.array([0, 0, 0.08]))

    com = sim.com()
    gz = sim.ground_height(com[0], com[1])
    self.com_handle.position = com
    self.com_ground_handle.position = np.array([com[0], com[1], gz + 0.005])
    scene.add_line_segments("/overlay/com_drop", np.array([[com, [com[0], com[1], gz]]]), colors=(230, 40, 40), thickness=0.004)
    poly = support_polygon(points)
    if poly is not None and len(poly) >= 2:
      pts3 = np.array([[p[0], p[1], sim.ground_height(p[0], p[1]) + 0.004] for p in poly])
      segs = np.stack([pts3, np.roll(pts3, -1, axis=0)], axis=1)
      inside = point_in_polygon(com[:2], poly)
      scene.add_line_segments("/overlay/support", segs, colors=(40, 200, 80) if inside else (230, 40, 40), thickness=0.008)
    else:
      scene.add_line_segments("/overlay/support", np.zeros((1, 2, 3)), colors=(0, 0, 0), visible=False)

    wrench = sim.applied_wrench()
    if np.linalg.norm(wrench[:3]) > 1e-6:
      scene.add_arrows("/overlay/wrench", np.array([[torso, torso + wrench[:3] * scale]]), colors=(200, 40, 200), shaft_radius=0.012, head_radius=0.03, head_length=0.06)
    else:
      scene.add_arrows("/overlay/wrench", np.zeros((1, 2, 3)), colors=(0, 0, 0), visible=False)

  # GUI.

  def _build_gui(self) -> None:
    gui = self.server.gui
    with gui.add_folder("Checkpoint"):
      self.ckpt_dropdown = gui.add_dropdown("checkpoint", options=["latest (auto)"], initial_value="latest (auto)")
      self.ckpt_status = gui.add_markdown("no checkpoint loaded")
      self.ckpt_dropdown.on_update(lambda _: self._on_ckpt_select())
    with gui.add_folder("Playback"):
      self.play_button = gui.add_button("Pause")
      self.step_button = gui.add_button("Step")
      self.reset_button = gui.add_button("Reset")
      self.speed = gui.add_slider("speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
      self.auto_reset = gui.add_checkbox("auto-reset on fall", initial_value=True)
      self.play_button.on_click(lambda _: self._toggle_play())
      self.step_button.on_click(lambda _: setattr(self, "step_request", True))
      self.reset_button.on_click(lambda _: setattr(self, "reset_request", True))
    with gui.add_folder("Command"):
      self.vx = gui.add_slider("vx [m/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
      self.vy = gui.add_slider("vy [m/s]", min=-0.5, max=0.5, step=0.05, initial_value=0.0)
      self.wz = gui.add_slider("wz [rad/s]", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
      zero = gui.add_button("Zero command")
      zero.on_click(lambda _: self._zero_command())
    with gui.add_folder("View"):
      self.follow = gui.add_checkbox("follow torso", initial_value=True)
      self.show_overlays = gui.add_checkbox("overlays", initial_value=True)
      self.show_collision = gui.add_checkbox("collision geoms", initial_value=False)
      self.noise = gui.add_slider("obs noise level", min=0.0, max=2.0, step=0.1, initial_value=self.sim.noise_level)
      self.show_collision.on_update(lambda _: self._toggle_collision())
    with gui.add_folder("Readout"):
      self.readout = gui.add_markdown("")
    with gui.add_folder("Randomization sample"):
      self.rand_text = gui.add_markdown(self._randomization_markdown())

  def _randomization_markdown(self) -> str:
    return "\n".join(f"- **{k}**: {v}" for k, v in self.sim.randomization_sample.items())

  def _toggle_play(self) -> None:
    self.playing = not self.playing
    self.play_button.label = "Pause" if self.playing else "Play"

  def _zero_command(self) -> None:
    self.vx.value, self.vy.value, self.wz.value = 0.0, 0.0, 0.0

  def _toggle_collision(self) -> None:
    for h in self.collision_handles.values():
      h.visible = self.show_collision.value

  def _on_ckpt_select(self) -> None:
    choice = self.ckpt_dropdown.value
    if choice == "latest (auto)":
      self.auto_ckpt = True
      self._load_latest()
    else:
      self.auto_ckpt = False
      self._load(os.path.join(self.ckpt_dir, choice))

  def _load(self, path: str) -> None:
    def work():
      try:
        pol = Policy(path)
      except Exception as e:  # noqa: BLE001
        self.ckpt_status.content = f"failed to load `{os.path.basename(path)}`: {e}"
        return
      with self.policy_lock:
        self.policy = pol
      self.ckpt_status.content = f"loaded step **{pol.step}** (`{os.path.basename(path)}`)"
    threading.Thread(target=work, daemon=True).start()

  def _load_latest(self) -> None:
    latest = checkpoints.latest_checkpoint(self.ckpt_dir)
    if latest and (self.policy is None or self.policy.path != latest):
      self._load(latest)

  def _poll_checkpoints(self) -> None:
    while True:
      names = [os.path.basename(p) for p in checkpoints.list_checkpoints(self.ckpt_dir)]
      options = ["latest (auto)"] + names[::-1]
      if list(self.ckpt_dropdown.options) != options:
        current = self.ckpt_dropdown.value
        self.ckpt_dropdown.options = options
        self.ckpt_dropdown.value = current if current in options else "latest (auto)"
      if self.auto_ckpt:
        self._load_latest()
      time.sleep(2.0)

  def _update_readout(self) -> None:
    sim = self.sim
    v = sim.local_linvel()
    g = sim.gyro()
    cmd = sim.command
    lin_err = float(np.linalg.norm(cmd[:2] - v[:2]))
    ang_err = float(abs(cmd[2] - g[2]))
    contact = sim.foot_contacts()
    self.readout.content = (
        f"step **{sim.step_count}** ({sim.step_count * sim.dt:.1f} s) | policy step {self.policy.step if self.policy else '-'}\n\n"
        f"cmd (vx, vy, wz) = ({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})\n\n"
        f"act (vx, vy, wz) = ({v[0]:+.2f}, {v[1]:+.2f}, {g[2]:+.2f})\n\n"
        f"lin err **{lin_err:.3f}** m/s | ang err **{ang_err:.3f}** rad/s\n\n"
        f"height {sim.d.qpos[2]:.3f} m | contacts L={'stance' if contact[0] else 'swing'} R={'stance' if contact[1] else 'swing'}"
        f" | air time L={sim.air_time[0]:.2f} R={sim.air_time[1]:.2f}\n\n"
        + ("**FALLEN**" if sim.fallen else "")
    )

  def _follow_cameras(self) -> None:
    target = self.sim.torso_pos()
    for client in self.server.get_clients().values():
      cid = client.client_id
      offset = self.client_offsets.get(cid, self.camera_offset0)
      pos = target + offset
      self.commanded_positions.setdefault(cid, collections.deque(maxlen=30)).append(pos)
      self.commanded_targets.setdefault(cid, collections.deque(maxlen=30)).append(target)
      client.camera.position = pos
      client.camera.look_at = target

  # Main loop.

  def run(self) -> None:
    frame = 0
    while True:
      t0 = time.perf_counter()
      self.sim.noise_level = self.noise.value
      self.sim.command = np.array([self.vx.value, self.vy.value, self.wz.value])
      if self.reset_request:
        self.sim.reset()
        self.reset_request = False
        self.fall_time = None
      if self.policy is not None and (self.playing or self.step_request):
        with self.policy_lock:
          action = self.policy(self.sim.obs(self.policy.obs_sizes))
        self.sim.step(action)
        self.step_request = False
        if self.sim.fallen and self.auto_reset.value:
          self.fall_time = self.fall_time or time.time()
          if time.time() - self.fall_time > 1.0:
            self.sim.reset()
            self.fall_time = None
      self._update_scene()
      if self.follow.value:
        self._follow_cameras()
      if frame % 5 == 0:
        self._update_readout()
      frame += 1
      elapsed = time.perf_counter() - t0
      time.sleep(max(0.0, self.sim.dt / self.speed.value - elapsed))


def main(argv=None) -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True)
  p.add_argument("--env", default=None, help="defaults to the env recorded in <run>/run_config.json")
  p.add_argument("--port", type=int, default=8080)
  p.add_argument("--noise-level", type=float, default=1.0)
  args = p.parse_args(argv)
  env_name = args.env
  cfg_path = os.path.join(args.run, "run_config.json")
  if env_name is None:
    env_name = json.load(open(cfg_path))["env"] if os.path.exists(cfg_path) else "G1JoystickFlat"
  Viewer(args.run, env_name, args.port, args.noise_level).run()


if __name__ == "__main__":
  main()
