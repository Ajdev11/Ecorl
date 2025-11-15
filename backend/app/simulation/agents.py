from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def baseline_myopic_markup(
	costs: np.ndarray,
	alpha: float,
	regulated_floor: Optional[float] = None,
	regulated_ceiling: Optional[float] = None,
) -> np.ndarray:
	"""
	Sets prices equal to cost + 1/alpha (constant markup under logit with myopic rule).
	Applies simple regulation bounds if provided.
	"""
	markup = 1.0 / alpha
	prices = costs + markup
	if regulated_floor is not None:
		prices = np.maximum(prices, regulated_floor)
	if regulated_ceiling is not None:
		prices = np.minimum(prices, regulated_ceiling)
	return prices


def epsilon_greedy_bandit_prices(
	costs: np.ndarray,
	markup_grid: np.ndarray,
	q_values: np.ndarray,
	action_counts: np.ndarray,
	epsilon: float,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Epsilon-greedy over a fixed markup grid. Each firm selects a markup action.
	Returns (prices, chosen_action_indices).
	"""
	num_firms, num_actions = q_values.shape
	assert markup_grid.shape[0] == num_actions

	# Exploration vs exploitation
	explore = rng.random(num_firms) < epsilon
	best_actions = np.argmax(q_values, axis=1)
	random_actions = rng.integers(low=0, high=num_actions, size=num_firms)
	actions = np.where(explore, random_actions, best_actions)

	markups = markup_grid[actions]
	prices = costs + markups
	return prices, actions


def _discretize_state(costs: np.ndarray, prev_prices: np.ndarray, num_bins: int) -> int:
	"""Discretize state from continuous costs/prices into a single integer."""
	# Simple state: mean cost and mean price quantized into bins
	mean_cost = np.mean(costs)
	mean_price = np.mean(prev_prices) if prev_prices is not None and len(prev_prices) > 0 else mean_cost
	
	# Quantize into bins (0..num_bins-1)
	cost_bin = min(int(mean_cost * num_bins / 5.0), num_bins - 1)
	price_bin = min(int(mean_price * num_bins / 5.0), num_bins - 1)
	# Combine into single state index
	state = cost_bin * num_bins + price_bin
	return min(state, num_bins * num_bins - 1)


def q_learning_prices(
	costs: np.ndarray,
	markup_grid: np.ndarray,
	q_table: np.ndarray,
	prev_state: Optional[int],
	prev_actions: Optional[np.ndarray],
	prev_rewards: Optional[np.ndarray],
	learning_rate: float,
	discount: float,
	epsilon: float,
	num_state_bins: int,
	rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
	"""
	Q-learning with tabular state-action values. State = discretized (mean_cost, mean_price).
	q_table shape: (num_states, num_firms, num_actions).
	Returns (prices, chosen_action_indices, new_state).
	"""
	num_states, num_firms, num_actions = q_table.shape
	assert markup_grid.shape[0] == num_actions
	
	# Discretize current state
	current_state = _discretize_state(costs, prev_actions, num_state_bins)
	current_state = min(current_state, num_states - 1)
	
	# Q-learning update if we have previous state/action/reward
	if prev_state is not None and prev_actions is not None and prev_rewards is not None:
		prev_state = min(prev_state, num_states - 1)
		for i in range(num_firms):
			a_prev = int(prev_actions[i])
			reward = prev_rewards[i]
			# Bellman update: Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
			max_q_next = np.max(q_table[current_state, i, :])
			q_table[prev_state, i, a_prev] += learning_rate * (reward + discount * max_q_next - q_table[prev_state, i, a_prev])
	
	# Epsilon-greedy action selection
	explore = rng.random(num_firms) < epsilon
	best_actions = np.array([np.argmax(q_table[current_state, i, :]) for i in range(num_firms)])
	random_actions = rng.integers(low=0, high=num_actions, size=num_firms)
	actions = np.where(explore, random_actions, best_actions)
	
	markups = markup_grid[actions]
	prices = costs + markups
	return prices, actions, current_state


def actor_critic_prices(
	costs: np.ndarray,
	markup_grid: np.ndarray,
	policy_probs: np.ndarray,
	value_table: np.ndarray,
	prev_state: Optional[int],
	prev_actions: Optional[np.ndarray],
	prev_rewards: Optional[np.ndarray],
	learning_rate_policy: float,
	learning_rate_value: float,
	discount: float,
	num_state_bins: int,
	rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
	"""
	Actor-critic with tabular policy and value function. Policy is softmax over actions.
	policy_probs shape: (num_states, num_firms, num_actions).
	value_table shape: (num_states, num_firms).
	Returns (prices, chosen_action_indices, new_state).
	"""
	num_states, num_firms, num_actions = policy_probs.shape
	assert markup_grid.shape[0] == num_actions
	
	current_state = _discretize_state(costs, prev_actions, num_state_bins)
	current_state = min(current_state, num_states - 1)
	
	# Update value function (critic) if we have previous state/reward
	if prev_state is not None and prev_rewards is not None:
		prev_state = min(prev_state, num_states - 1)
		for i in range(num_firms):
			reward = prev_rewards[i]
			td_error = reward + discount * value_table[current_state, i] - value_table[prev_state, i]
			value_table[prev_state, i] += learning_rate_value * td_error
			
			# Update policy (actor) using policy gradient
			if prev_actions is not None:
				a_prev = int(prev_actions[i])
				# REINFORCE-style update: increase probability of action that led to positive TD error
				for a in range(num_actions):
					indicator = 1.0 if a == a_prev else 0.0
					policy_probs[prev_state, i, a] += learning_rate_policy * td_error * (indicator - policy_probs[prev_state, i, a])
				# Renormalize to stay valid probability distribution
				policy_probs[prev_state, i, :] = np.maximum(policy_probs[prev_state, i, :], 1e-8)
				policy_probs[prev_state, i, :] /= policy_probs[prev_state, i, :].sum()
	
	# Sample actions from current policy (softmax)
	actions = np.zeros(num_firms, dtype=int)
	for i in range(num_firms):
		probs = policy_probs[current_state, i, :]
		actions[i] = rng.choice(num_actions, p=probs)
	
	markups = markup_grid[actions]
	prices = costs + markups
	return prices, actions, current_state

