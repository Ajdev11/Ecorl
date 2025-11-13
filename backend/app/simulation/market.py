from __future__ import annotations

from typing import Tuple
import numpy as np


def logit_shares(
	prices: np.ndarray,
	qualities: np.ndarray,
	alpha: float,
) -> np.ndarray:
	"""
	Compute multinomial logit market shares with an outside option.
	u_i = quality_i - alpha * price_i
	share_i = exp(u_i) / (1 + sum_j exp(u_j))
	share_0 (outside) = 1 / (1 + sum_j exp(u_j))
	"""
	utilities = qualities - alpha * prices
	max_u = np.max(utilities)
	exp_u = np.exp(utilities - max_u)  # numerical stability
	denom = 1.0 + np.sum(exp_u)
	return exp_u / denom


def approximate_consumer_surplus_per_consumer(
	prices: np.ndarray,
	qualities: np.ndarray,
	alpha: float,
) -> float:
	"""
	For i.i.d. type-1 extreme value errors with scale 1, per-period expected CS is:
	CS = (1/alpha) * log(1 + sum_j exp(quality_j - alpha * price_j))
	We return the value for a single period; callers should average over time.
	"""
	utilities = qualities - alpha * prices
	max_u = np.max(utilities)
	logsum = max_u + np.log(1.0 + np.sum(np.exp(utilities - max_u)))
	return (1.0 / alpha) * logsum


def evolve_costs_random_walk(
	prev_costs: np.ndarray,
	sigma: float,
	rng: np.random.Generator,
) -> np.ndarray:
	innov = rng.normal(loc=0.0, scale=sigma, size=prev_costs.shape)
	costs = prev_costs + innov
	return np.clip(costs, a_min=0.0, a_max=None)

