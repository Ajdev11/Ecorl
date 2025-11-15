"""Compare policies across runs."""
import pandas as pd
from pathlib import Path

def main():
    metrics_path = Path("data/aggregates/metrics.csv")
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run: python scripts/aggregate_runs.py")
        return
    
    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} runs")
    print("\n" + "="*60)
    print("POLICY COMPARISON (Mean Values)")
    print("="*60)
    
    if "policy" in df.columns:
        comparison = df.groupby("policy")[
            ["mean_price", "total_welfare", "consumer_surplus_per_consumer", "cross_corr_lag1_f0_f1"]
        ].mean().round(4)
        print(comparison)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS (Mean ± Std, Count)")
        print("="*60)
        summary = df.groupby("policy")[["mean_price", "total_welfare"]].agg(["mean", "std", "count"]).round(4)
        print(summary)
        
        print("\n" + "="*60)
        print("COORDINATION METRICS")
        print("="*60)
        coord = df.groupby("policy")[
            ["cross_corr_lag1_f0_f1", "identical_price_share", "within_var_mean"]
        ].mean().round(4)
        print(coord)
    else:
        print("No 'policy' column found in metrics.csv")
        print("\nAvailable columns:", df.columns.tolist())

if __name__ == "__main__":
    main()

