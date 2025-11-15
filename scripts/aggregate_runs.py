"""
Aggregate per-run summaries and coordination metrics into a single CSV.

Usage (from project root):
  python scripts/aggregate_runs.py
  python scripts/aggregate_runs.py --runs <run_id1> <run_id2>
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
OUT_DIR = PROJECT_ROOT / "data" / "aggregates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
	return json.loads(path.read_text(encoding="utf-8"))


def read_panel(run_dir: Path) -> pd.DataFrame:
	parquet = run_dir / "panel.parquet"
	csv = run_dir / "panel.csv"
	if parquet.exists():
		return pd.read_parquet(parquet)
	if csv.exists():
		return pd.read_csv(csv)
	raise FileNotFoundError(f"No panel file in {run_dir}")


def coordination_metrics(panel: pd.DataFrame) -> Dict[str, float]:
	wide = panel.pivot(index="t", columns="firm_id", values="price").sort_index(axis=1)
	identical_price_share = float((wide.nunique(axis=1) == 1).mean())
	within_var_mean = float(wide.var(axis=1).mean())
	try:
		cross_corr_lag1 = float(wide[0].shift(1).corr(wide[1]))
	except Exception:
		cross_corr_lag1 = float("nan")
	mean_price_series = wide.mean(axis=1)
	threshold = float(mean_price_series.quantile(0.9))
	streak = 0
	max_streak = 0
	for v in (mean_price_series > threshold).astype(int):
		if v:
			streak += 1
			max_streak = max(max_streak, streak)
		else:
			streak = 0
	return {
		"identical_price_share": identical_price_share,
		"within_var_mean": within_var_mean,
		"cross_corr_lag1_f0_f1": cross_corr_lag1,
		"max_high_price_streak": int(max_streak),
	}


def collect_run(run_id: str) -> Optional[Dict]:
	run_dir = RUNS_DIR / run_id
	if not run_dir.exists():
		return None
	summary_path = run_dir / "summary.json"
	if not summary_path.exists():
		return None
	summary = load_json(summary_path)
	status_path = run_dir / "status.json"
	status = load_json(status_path) if status_path.exists() else {"status": "unknown"}
	scenario_path = run_dir / "scenario.json"
	scenario = load_json(scenario_path) if scenario_path.exists() else {}
	config_path = run_dir / "config.json"
	config = load_json(config_path) if config_path.exists() else {}
	panel = read_panel(run_dir)
	metrics = coordination_metrics(panel)
	row = {
		"run_id": run_id,
		"status": status.get("status", "unknown"),
		"name": scenario.get("name"),
		"num_firms": scenario.get("num_firms"),
		"num_consumers": scenario.get("num_consumers"),
		"time_periods": scenario.get("time_periods"),
		"policy": scenario.get("policy"),
		"policy_params": json.dumps(scenario.get("policy_params", {})),
		"regulation": json.dumps(scenario.get("regulation", {})),
		"seed": config.get("seed"),
	}
	row.update(summary)
	row.update(metrics)
	return row


def find_all_runs() -> List[str]:
	return sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="*", help="Specific run IDs to aggregate")
	parser.add_argument("--out", default=str(OUT_DIR / "metrics.csv"), help="Output CSV path")
	args = parser.parse_args()

	run_ids = args.runs if args.runs else find_all_runs()
	rows: List[Dict] = []
	for rid in run_ids:
		row = collect_run(rid)
		if row is not None:
			rows.append(row)
	if not rows:
		print("No runs found to aggregate.")
		return
	df = pd.DataFrame(rows)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)
	print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
	main()

