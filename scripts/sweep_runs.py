"""
Launch a small sweep of experiments via the running API and aggregate results.

Usage:
  # Default sweep: myopic vs bandit, seeds 51-55, no regulation
  python scripts/sweep_runs.py

  # With binding price ceiling
  python scripts/sweep_runs.py --ceiling 2.6
"""
from __future__ import annotations

import time
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path

import requests
import pandas as pd


API = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_run_payload(policy: str, seed: int, ceiling: Optional[float]) -> Dict:
	regulation = None if ceiling is None else {"price_floor": 0, "price_ceiling": float(ceiling)}
	return {
		"scenario": {
			"name": f"{policy}-{'cap' if regulation else 'nocap'}",
			"num_firms": 3,
			"num_consumers": 10000,
			"time_periods": 100,
			"demand": {"price_sensitivity": 0.5, "mean_quality": 1.0},
			"cost_process": {"initial_cost": 1.0, "sigma": 0.1},
			"regulation": regulation,
			"policy": policy,
			"policy_params": {"epsilon": 0.1, "markup_min": 0.2, "markup_max": 3.0, "num_actions": 15} if policy == "epsilon_bandit" else {},
		},
		"config": {"seed": int(seed), "log_interval": 10},
	}


def launch_and_wait(payload: Dict) -> Dict:
	run_id = requests.post(f"{API}/runs", json=payload).json()["run_id"]
	while True:
		resp = requests.get(f"{API}/runs/{run_id}").json()
		if resp["status"] == "succeeded":
			break
		time.sleep(0.5)
	summary = requests.get(f"{API}/results/{run_id}").json()
	return {"run_id": run_id, **summary, "policy": payload["scenario"]["policy"], "seed": payload["config"]["seed"], "regulation": payload["scenario"]["regulation"]}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--seeds", nargs="*", type=int, default=[51, 52, 53, 54, 55])
	parser.add_argument("--ceiling", type=float, default=None, help="Set a price ceiling to make regulation binding (e.g., 2.6)")
	parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "aggregates" / "sweep_results.csv"))
	args = parser.parse_args()

	policies = ["myopic", "epsilon_bandit"]
	results: List[Dict] = []
	for seed in args.seeds:
		for policy in policies:
			payload = create_run_payload(policy=policy, seed=seed, ceiling=args.ceiling)
			results.append(launch_and_wait(payload))

	df = pd.DataFrame(results)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)
	print(f"Wrote sweep results to {out_path}")


if __name__ == "__main__":
	main()

