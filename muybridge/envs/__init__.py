"""Environment registry."""

from typing import Any, Dict, Optional

from ml_collections import config_dict

from muybridge.envs import g1_joystick

ENVS = {
    "G1JoystickFlat": lambda cfg, ov: g1_joystick.G1Joystick("flat", cfg, ov),
    "G1JoystickRough": lambda cfg, ov: g1_joystick.G1Joystick("rough", cfg, ov),
}


def default_config(name: str) -> config_dict.ConfigDict:
  if name not in ENVS:
    raise KeyError(f"unknown env {name}; known: {sorted(ENVS)}")
  return g1_joystick.default_config()


def make(
    name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
  if config is None:
    config = default_config(name)
  return ENVS[name](config, config_overrides)
