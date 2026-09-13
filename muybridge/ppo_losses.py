"""PPO loss forked from ``brax.training.agents.ppo.losses`` (Brax 0.14.2, Apache 2.0).

Identical to upstream except that the metrics dict also reports
``clip_fraction``, ``explained_variance`` and raw ``entropy``. ``train.py``
installs this module in place of Brax's via
``brax.training.agents.ppo.train.ppo_losses = muybridge.ppo_losses``.
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.ppo import losses as brax_losses
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp

PPONetworkParams = brax_losses.PPONetworkParams
compute_gae = brax_losses.compute_gae
quantile_huber_loss = brax_losses.quantile_huber_loss


def compute_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
    use_distributional_critic: bool = False,
) -> Tuple[jnp.ndarray, types.Metrics]:
  parametric_action_distribution = ppo_network.parametric_action_distribution
  policy_apply = ppo_network.policy_network.apply
  value_apply = ppo_network.value_network.apply

  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  policy_logits = policy_apply(normalizer_params, params.policy, data.observation)

  if use_distributional_critic:
    baseline, baseline_quantiles = value_apply(normalizer_params, params.value, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value, _ = value_apply(normalizer_params, params.value, terminal_obs)
  else:
    baseline = value_apply(normalizer_params, params.value, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)
    baseline_quantiles = None

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.extras['policy_extras']['raw_action']
  )
  behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting,
  )
  gae_returns = jax.lax.stop_gradient(jnp.add(advantages, jax.lax.stop_gradient(baseline)))
  if normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
  clip_fraction = jnp.mean((jnp.abs(rho_s - 1.0) > clipping_epsilon).astype(jnp.float32))

  if use_distributional_critic:
    v_loss = quantile_huber_loss(baseline_quantiles, gae_returns, kappa=clipping_epsilon_value) * vf_coefficient
  else:
    v_error = vs - baseline
    v_loss = v_error * v_error
    if clipping_epsilon_value is not None:
      old_values = data.extras['policy_extras']['value']
      v_clipped = old_values + jnp.clip(baseline - old_values, -clipping_epsilon_value, clipping_epsilon_value)
      v_loss_clipped = (vs - v_clipped) ** 2
      v_loss = jnp.maximum(v_loss, v_loss_clipped)
    v_loss = jnp.mean(v_loss) * 0.5 * vf_coefficient

  explained_variance = 1.0 - jnp.var(vs - baseline) / (jnp.var(vs) + 1e-8)

  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy
  total_loss = policy_loss + v_loss + entropy_loss

  new_dist = parametric_action_distribution.create_dist(policy_logits)
  if hasattr(new_dist, 'kl_divergence'):
    old_dist = parametric_action_distribution.create_dist(data.extras['policy_extras']['distribution_params'])
    kl = jnp.mean(new_dist.kl_divergence(old_dist))
  else:
    kl = jnp.array(0.0)

  return total_loss, {
      'total_loss': total_loss,
      'policy_loss': policy_loss,
      'v_loss': v_loss,
      'entropy_loss': entropy_loss,
      'entropy': entropy,
      'kl_mean': kl,
      'clip_fraction': clip_fraction,
      'explained_variance': explained_variance,
      'policy_dist_mean_std': jnp.mean(new_dist.scale),
      'policy_dist_max_std': jnp.max(new_dist.scale),
      'policy_dist_min_std': jnp.min(new_dist.scale),
      'policy_dist_mean_loc': jnp.mean(new_dist.loc),
      'policy_dist_max_loc': jnp.max(new_dist.loc),
      'policy_dist_min_loc': jnp.min(new_dist.loc),
  }
