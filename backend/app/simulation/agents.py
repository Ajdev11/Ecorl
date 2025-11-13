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

