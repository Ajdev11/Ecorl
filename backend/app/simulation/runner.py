from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
try:
	import pandas as pd  # type: ignore[import-not-found]
except Exception as _e:
	# Pandas is required to run simulations; provide a clear message if missing
	raise RuntimeError("pandas is required to run the simulation. Install with: python -m pip install pandas") from _e
try:
	import ray  # type: ignore[import-not-found]
	_RAY_AVAILABLE = True
except Exception:
	_RAY_AVAILABLE = False
	class _RemoteWrapper:
		def __init__(self, fn):
			self.fn = fn
		def remote(self, *args, **kwargs):
			# Synchronous fallback if Ray is not installed
			return self.fn(*args, **kwargs)
	class _RayStub:
		@staticmethod
		def remote(fn):
			return _RemoteWrapper(fn)
	ray = _RayStub()  # type: ignore[assignment]
try:
	import mlflow  # type: ignore[import-not-found]
except Exception:
	class _NoopRun:
		def __enter__(self):  # noqa: D401
			return self
		def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
			return False
	class _MlflowNoop:
		@staticmethod
		def set_tracking_uri(*args, **kwargs) -> None:
			return None
		@staticmethod
		def set_experiment(*args, **kwargs) -> None:
			return None
		@staticmethod
		def start_run(*args, **kwargs):
			return _NoopRun()
		@staticmethod
		def log_metrics(*args, **kwargs) -> None:
			return None
		@staticmethod
		def log_artifact(*args, **kwargs) -> None:
			return None
	mlflow = _MlflowNoop()  # type: ignore[assignment]

from ..schemas import Scenario, RunConfig
from .market import (
	logit_shares,
	evolve_costs_random_walk,
	approximate_consumer_surplus_per_consumer,
)
from .agents import baseline_myopic_markup
from ..utils.io import ensure_dir, write_json, write_parquet, append_log_line
from ..utils.run_store import set_status


def _simulate_once(
	num_firms: int,
	num_consumers: int,
	time_periods: int,
	alpha: float,
	mean_quality: float,
	initial_cost: float,
	cost_sigma: float,
	price_floor: float | None,
	price_ceiling: float | None,
	log_path: Path,
	rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
	qualities = np.full(shape=(num_firms,), fill_value=mean_quality, dtype=float)
	costs = np.full(shape=(num_firms,), fill_value=initial_cost, dtype=float)

	records = []
	cs_values = []

	for t in range(time_periods):
		# Firms choose prices via myopic markup rule (baseline)
		prices = baseline_myopic_markup(costs=costs, alpha=alpha, regulated_floor=price_floor, regulated_ceiling=price_ceiling)
		shares = logit_shares(prices=prices, qualities=qualities, alpha=alpha)
		quantities = num_consumers * shares
		profits = (prices - costs) * quantities

		cs_t = approximate_consumer_surplus_per_consumer(prices=prices, qualities=qualities, alpha=alpha)
		cs_values.append(cs_t)

		for i in range(num_firms):
			records.append(
				{
					"t": t,
					"firm_id": i,
					"price": float(prices[i]),
					"cost": float(costs[i]),
					"quantity": float(quantities[i]),
					"profit": float(profits[i]),
				}
			)

		if (t + 1) % 10 == 0 or t == time_periods - 1:
			append_log_line(log_path, f"t={t+1}/{time_periods}  mean_price={prices.mean():.4f}")

		# Evolve costs (exogenous process)
		costs = evolve_costs_random_walk(prev_costs=costs, sigma=cost_sigma, rng=rng)

	panel = pd.DataFrame.from_records(records)
	mean_price = float(panel["price"].mean())
	mean_profit_per_firm = float(panel.groupby("firm_id")["profit"].mean().mean())
	cs_per_consumer = float(np.mean(cs_values))
	total_welfare = cs_per_consumer * num_consumers + float(panel["profit"].sum())
	summary = {
		"mean_price": mean_price,
		"mean_profit_per_firm": mean_profit_per_firm,
		"consumer_surplus_per_consumer": cs_per_consumer,
		"total_welfare": total_welfare,
	}
	return panel, summary


@ray.remote
def ray_simulation_task(
	run_id: str,
	scenario_dict: Dict,
	run_config_dict: Dict,
	base_dir_str: str,
) -> Dict:
	base_dir = Path(base_dir_str)
	run_dir = base_dir / "data" / "runs" / run_id
	ensure_dir(run_dir)
	log_path = run_dir / "stdout.log"
	status_path = run_dir / "status.json"

	try:
		# Update status: running
		set_status(status_path, "running", "Simulation started")
		append_log_line(log_path, f"Run {run_id} started")

		scenario = Scenario(**scenario_dict)
		run_config = RunConfig(**run_config_dict)

		rng = np.random.default_rng(run_config.seed)

		mlflow.set_tracking_uri("file:./mlruns")
		mlflow.set_experiment("algorithmic-competition")
		with mlflow.start_run(run_name=run_id):
			panel, summary = _simulate_once(
				num_firms=scenario.num_firms,
				num_consumers=scenario.num_consumers,
				time_periods=scenario.time_periods,
				alpha=scenario.demand.price_sensitivity,
				mean_quality=scenario.demand.mean_quality,
				initial_cost=scenario.cost_process.initial_cost,
				cost_sigma=scenario.cost_process.sigma,
				price_floor=(scenario.regulation.price_floor if scenario.regulation else None),
				price_ceiling=(scenario.regulation.price_ceiling if scenario.regulation else None),
				log_path=log_path,
				rng=rng,
			)

			# Save artifacts
			write_parquet(panel, run_dir / "panel.parquet")
			write_json(run_dir / "summary.json", summary)
			try:
				mlflow.log_metrics(
					{
						"mean_price": summary["mean_price"],
						"mean_profit_per_firm": summary["mean_profit_per_firm"],
						"consumer_surplus_per_consumer": summary["consumer_surplus_per_consumer"],
						"total_welfare": summary["total_welfare"],
					}
				)
				mlflow.log_artifact(str(run_dir / "summary.json"))
				mlflow.log_artifact(str(run_dir / "panel.parquet"))
			except Exception:
				# No-op if mlflow is stubbed or unavailable
				pass

		set_status(status_path, "succeeded", "Simulation completed")
		append_log_line(log_path, "Run completed successfully")
		return summary
	except Exception as e:
		set_status(status_path, "failed", f"{e}")
		append_log_line(log_path, f"Run failed: {e}")
		raise

