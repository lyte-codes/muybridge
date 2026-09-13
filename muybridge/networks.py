"""Actor/critic network factory shared by training, eval and the viewer."""

import functools

from brax.training.agents.ppo import networks as ppo_networks

POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)


def make_network_factory(policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN):
  return functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=tuple(policy_hidden),
      value_hidden_layer_sizes=tuple(value_hidden),
      policy_obs_key="state",
      value_obs_key="privileged_state",
  )
