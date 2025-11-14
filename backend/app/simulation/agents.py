from __future__ import annotations

from typing import Optional
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

