"""Unitree G1 model loading.

The upstream MuJoCo Playground / Menagerie XML is never edited. All structural
changes happen at runtime on an `mujoco.MjSpec`:

* Arm joints are fused into the torso at a fixed pose. Arm links keep their
  mass and inertia, so the torso carries the same load as the real robot, but
  no arm degrees of freedom are simulated or exposed to the policy.
* The remaining actuators are the 12 leg joints plus the 3 waist joints, which
  is the interface contract for the locomotion policy.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Sequence

import mujoco
import numpy as np

from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts

ARM_KEYWORDS = ("shoulder", "elbow", "wrist")
LEG_JOINT_NAMES = tuple(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES
TERRAIN_XML = {
    "flat": consts.FEET_ONLY_FLAT_TERRAIN_XML,
    "rough": consts.FEET_ONLY_ROUGH_TERRAIN_XML,
}


def is_arm_name(name: str) -> bool:
  return any(k in name for k in ARM_KEYWORDS)


@dataclasses.dataclass(frozen=True)
class G1Model:
  mj_model: mujoco.MjModel
  spec: mujoco.MjSpec
  assets: Dict[str, bytes]
  terrain: str
  fused_arms: bool
  xml_path: str

  @property
  def policy_actuator_names(self) -> Sequence[str]:
    return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]


def _quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), float(angle))
  return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
  return out


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  out = np.zeros(3)
  mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
  return out


def fuse_arm_joints(spec: mujoco.MjSpec, pose_key: str) -> None:
  """Delete every arm joint and actuator, baking the pose from `pose_key` into body frames.

  A hinge joint at local point c with axis a rotated by theta maps body-local
  points x -> c + R (x - c). Folding that into the body's frame relative to its
  parent gives pos' = pos + q * (c - R c) and quat' = q * quat(a, theta).
  """
  full = spec.compile()
  key_qpos = np.array(full.key(pose_key).qpos)
  arm_joints = [j.name for j in spec.joints if is_arm_name(j.name)]
  arm_qpos_adr = []
  for name in arm_joints:
    jid = full.joint(name).id
    if full.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"expected hinge arm joint, got {name}")
    qadr = int(full.jnt_qposadr[jid])
    arm_qpos_adr.append(qadr)
    theta = key_qpos[qadr]
    joint = spec.joint(name)
    body = joint.parent
    q_joint = _quat_axis_angle(np.array(joint.axis), theta)
    c = np.array(joint.pos)
    shift = c - _rotate(q_joint, c)
    body.pos = np.array(body.pos) + _rotate(np.array(body.quat), shift)
    body.quat = _quat_mul(np.array(body.quat), q_joint)
    spec.delete(joint)
  arm_ctrl_ids = []
  for act in list(spec.actuators):
    if is_arm_name(act.name):
      arm_ctrl_ids.append(full.actuator(act.name).id)
      spec.delete(act)
  keep_q = np.setdiff1d(np.arange(full.nq), arm_qpos_adr)
  keep_u = np.setdiff1d(np.arange(full.nu), arm_ctrl_ids)
  for key in spec.keys:
    key.qpos = list(np.array(key.qpos)[keep_q])
    key.ctrl = list(np.array(key.ctrl)[keep_u])


def load_g1(
    terrain: str = "flat",
    fuse_arms: bool = True,
    pose_key: str = "knees_bent",
    sim_dt: float | None = None,
) -> G1Model:
  xml_path = TERRAIN_XML[terrain]
  assets = g1_base.get_assets()
  spec = mujoco.MjSpec.from_string(xml_path.read_text(), include=assets, assets=assets)
  if fuse_arms:
    fuse_arm_joints(spec, pose_key)
  mj_model = spec.compile()
  if sim_dt is not None:
    mj_model.opt.timestep = sim_dt
  if fuse_arms:
    expected = list(POLICY_JOINT_NAMES)
    actual = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    if actual != expected:
      raise RuntimeError(f"actuator order mismatch: {actual}")
  return G1Model(mj_model, spec, assets, terrain, fuse_arms, xml_path.as_posix())
