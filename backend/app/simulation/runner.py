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
from .agents import (
	baseline_myopic_markup,
	epsilon_greedy_bandit_prices,
	q_learning_prices,
	actor_critic_prices,
)
from .regulator import apply_price_bounds
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
	policy: str,
	policy_params: dict,
	log_path: Path,
	rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
	qualities = np.full(shape=(num_firms,), fill_value=mean_quality, dtype=float)
	costs = np.full(shape=(num_firms,), fill_value=initial_cost, dtype=float)

	records = []
	cs_values = []

	# Setup for learning policies
	needs_grid = policy in ("epsilon_bandit", "q_learning", "actor_critic")
	if needs_grid:
		num_actions = int(policy_params.get("num_actions", 15))
		markup_min = float(policy_params.get("markup_min", 0.2))
		markup_max = float(policy_params.get("markup_max", 3.0))
		markup_grid = np.linspace(markup_min, markup_max, num_actions, dtype=float)
	else:
		markup_grid = None  # type: ignore
		num_actions = 0

	# Bandit
	if policy == "epsilon_bandit":
		epsilon = float(policy_params.get("epsilon", 0.1))
		q_values = np.zeros((num_firms, num_actions), dtype=float)
		action_counts = np.zeros((num_firms, num_actions), dtype=float)
	else:
		q_values = None  # type: ignore
		action_counts = None  # type: ignore

	# Q-learning
	if policy == "q_learning":
		epsilon = float(policy_params.get("epsilon", 0.1))
		learning_rate = float(policy_params.get("learning_rate", 0.1))
		discount = float(policy_params.get("discount", 0.95))
		num_state_bins = int(policy_params.get("num_state_bins", 10))
		num_states = num_state_bins * num_state_bins
		q_table = np.zeros((num_states, num_firms, num_actions), dtype=float)
		prev_state = None
		prev_actions = None
		prev_rewards = None
	else:
		q_table = None  # type: ignore
		prev_state = None
		prev_actions = None
		prev_rewards = None
		learning_rate = 0.0  # unused
		discount = 0.0  # unused
		num_state_bins = 0  # unused

	# Actor-critic
	if policy == "actor_critic":
		learning_rate_policy = float(policy_params.get("learning_rate_policy", 0.01))
		learning_rate_value = float(policy_params.get("learning_rate_value", 0.1))
		discount = float(policy_params.get("discount", 0.95))
		num_state_bins = int(policy_params.get("num_state_bins", 10))
		num_states = num_state_bins * num_state_bins
		# Initialize uniform policy and zero values
		policy_probs = np.ones((num_states, num_firms, num_actions), dtype=float) / num_actions
		value_table = np.zeros((num_states, num_firms), dtype=float)
		prev_state = None
		prev_actions = None
		prev_rewards = None
	else:
		policy_probs = None  # type: ignore
		value_table = None  # type: ignore
		learning_rate_policy = 0.0  # unused
		learning_rate_value = 0.0  # unused

	for t in range(time_periods):
		# Firms choose prices by selected policy
		if policy == "myopic":
			prices = baseline_myopic_markup(costs=costs, alpha=alpha, regulated_floor=price_floor, regulated_ceiling=price_ceiling)
			chosen_actions = None
			current_state = None
		elif policy == "epsilon_bandit":
			prices, chosen_actions = epsilon_greedy_bandit_prices(
				costs=costs,
				markup_grid=markup_grid,  # type: ignore
				q_values=q_values,  # type: ignore
				action_counts=action_counts,  # type: ignore
				epsilon=epsilon,
				rng=rng,
			)
			prices = apply_price_bounds(prices, price_floor, price_ceiling)
			current_state = None
		elif policy == "q_learning":
			prices, chosen_actions, current_state = q_learning_prices(
				costs=costs,
				markup_grid=markup_grid,  # type: ignore
				q_table=q_table,  # type: ignore
				prev_state=prev_state,
				prev_actions=prev_actions,
				prev_rewards=prev_rewards,
				learning_rate=learning_rate,
				discount=discount,
				epsilon=epsilon,
				num_state_bins=num_state_bins,
				rng=rng,
			)
			prices = apply_price_bounds(prices, price_floor, price_ceiling)
		elif policy == "actor_critic":
			prices, chosen_actions, current_state = actor_critic_prices(
				costs=costs,
				markup_grid=markup_grid,  # type: ignore
				policy_probs=policy_probs,  # type: ignore
				value_table=value_table,  # type: ignore
				prev_state=prev_state,
				prev_actions=prev_actions,
				prev_rewards=prev_rewards,
				learning_rate_policy=learning_rate_policy,
				learning_rate_value=learning_rate_value,
				discount=discount,
				num_state_bins=num_state_bins,
				rng=rng,
			)
			prices = apply_price_bounds(prices, price_floor, price_ceiling)
		else:
			append_log_line(log_path, f"Unknown policy '{policy}', defaulting to myopic")
			prices = baseline_myopic_markup(costs=costs, alpha=alpha, regulated_floor=price_floor, regulated_ceiling=price_ceiling)
			chosen_actions = None
			current_state = None

		shares = logit_shares(prices=prices, qualities=qualities, alpha=alpha)
		quantities = num_consumers * shares
		profits = (prices - costs) * quantities

		cs_t = approximate_consumer_surplus_per_consumer(prices=prices, qualities=qualities, alpha=alpha)
		cs_values.append(cs_t)

		# Learning updates
		if policy == "epsilon_bandit":
			for i in range(num_firms):
				a = int(chosen_actions[i])  # type: ignore[index]
				action_counts[i, a] += 1.0  # type: ignore[index]
				q_values[i, a] += (profits[i] - q_values[i, a]) / action_counts[i, a]  # type: ignore[index]
		elif policy == "q_learning":
			# Q-learning updates are done inside q_learning_prices; store for next iteration
			prev_state = current_state
			prev_actions = chosen_actions  # type: ignore[assignment]
			prev_rewards = profits
		elif policy == "actor_critic":
			# Actor-critic updates are done inside actor_critic_prices; store for next iteration
			prev_state = current_state
			prev_actions = chosen_actions  # type: ignore[assignment]
			prev_rewards = profits

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
	scenario_path = run_dir / "scenario.json"
	config_path = run_dir / "config.json"

	try:
		# Update status: running
		set_status(status_path, "running", "Simulation started")
		append_log_line(log_path, f"Run {run_id} started")

		scenario = Scenario(**scenario_dict)
		run_config = RunConfig(**run_config_dict)
		# Persist inputs for aggregation/repro
		write_json(scenario_path, scenario.model_dump())
		write_json(config_path, run_config.model_dump())

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
			policy=scenario.policy,
			policy_params=scenario.policy_params,
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

