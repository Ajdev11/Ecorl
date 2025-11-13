from __future__ import annotations

from typing import Optional
import numpy as np


def apply_price_bounds(
	prices: np.ndarray,
	price_floor: Optional[float],
	price_ceiling: Optional[float],
) -> np.ndarray:
	if price_floor is not None:
		prices = np.maximum(prices, price_floor)
	if price_ceiling is not None:
		prices = np.minimum(prices, price_ceiling)
	return prices

