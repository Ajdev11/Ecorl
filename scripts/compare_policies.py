"""Compare policies across runs with visualizations."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

def main():
    metrics_path = Path("data/aggregates/metrics.csv")
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run: python scripts/aggregate_runs.py")
        return
    
    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} runs")
    
    # Filter out runs without policy
    df = df[df["policy"].notna()].copy()
    
    if len(df) == 0:
        print("No runs with policy information found.")
        return
    
    # Create figures directory
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("POLICY COMPARISON (Mean Values)")
    print("="*60)
    
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
    
    # Generate graphs
    print("\n" + "="*60)
    print("GENERATING GRAPHS...")
    print("="*60)
    
    # 1. Welfare Comparison: Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.boxplot(data=df, x="policy", y="mean_price", ax=axes[0])
    axes[0].set_title("Mean Price by Policy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Policy", fontsize=11)
    axes[0].set_ylabel("Mean Price", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    sns.boxplot(data=df, x="policy", y="consumer_surplus_per_consumer", ax=axes[1])
    axes[1].set_title("Consumer Surplus per Consumer by Policy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Policy", fontsize=11)
    axes[1].set_ylabel("Consumer Surplus per Consumer", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    sns.boxplot(data=df, x="policy", y="total_welfare", ax=axes[2])
    axes[2].set_title("Total Welfare by Policy", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Policy", fontsize=11)
    axes[2].set_ylabel("Total Welfare", fontsize=11)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "welfare_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'welfare_comparison.png'}")
    plt.close()
    
    # 2. Coordination Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.boxplot(data=df, x="policy", y="cross_corr_lag1_f0_f1", ax=axes[0])
    axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5, label="No correlation")
    axes[0].set_title("Price Cross-Correlation (Lag 1)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Policy", fontsize=11)
    axes[0].set_ylabel("Cross-Correlation", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    sns.boxplot(data=df, x="policy", y="identical_price_share", ax=axes[1])
    axes[1].set_title("Identical Price Share", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Policy", fontsize=11)
    axes[1].set_ylabel("Identical Price Share", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    sns.boxplot(data=df, x="policy", y="within_var_mean", ax=axes[2])
    axes[2].set_title("Within-Period Price Variance (Mean)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Policy", fontsize=11)
    axes[2].set_ylabel("Price Variance", fontsize=11)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "coordination_metrics.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'coordination_metrics.png'}")
    plt.close()
    
    # 3. Bar Charts: Mean Values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mean_values = df.groupby("policy")["mean_price"].mean().sort_values(ascending=False)
    mean_values.plot(kind="bar", ax=axes[0, 0], color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Mean Price by Policy (Average)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Policy", fontsize=11)
    axes[0, 0].set_ylabel("Mean Price", fontsize=11)
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    
    mean_values = df.groupby("policy")["total_welfare"].mean().sort_values(ascending=False)
    mean_values.plot(kind="bar", ax=axes[0, 1], color="forestgreen", edgecolor="black")
    axes[0, 1].set_title("Total Welfare by Policy (Average)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Policy", fontsize=11)
    axes[0, 1].set_ylabel("Total Welfare", fontsize=11)
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    mean_values = df.groupby("policy")["consumer_surplus_per_consumer"].mean().sort_values(ascending=False)
    mean_values.plot(kind="bar", ax=axes[1, 0], color="coral", edgecolor="black")
    axes[1, 0].set_title("Consumer Surplus by Policy (Average)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Policy", fontsize=11)
    axes[1, 0].set_ylabel("Consumer Surplus per Consumer", fontsize=11)
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    
    mean_values = df.groupby("policy")["cross_corr_lag1_f0_f1"].mean().sort_values(ascending=False)
    mean_values.plot(kind="bar", ax=axes[1, 1], color="purple", edgecolor="black")
    axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Price Cross-Correlation by Policy (Average)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Policy", fontsize=11)
    axes[1, 1].set_ylabel("Cross-Correlation", fontsize=11)
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_comparison_bars.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'mean_comparison_bars.png'}")
    plt.close()
    
    # 4. Scatter: Price vs Welfare
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for policy in df["policy"].unique():
        subset = df[df["policy"] == policy]
        ax.scatter(subset["mean_price"], subset["total_welfare"], label=policy, alpha=0.7, s=100)
    
    ax.set_xlabel("Mean Price", fontsize=11)
    ax.set_ylabel("Total Welfare", fontsize=11)
    ax.set_title("Price vs Total Welfare by Policy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "price_vs_welfare.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'price_vs_welfare.png'}")
    plt.close()
    
    # 5. Scatter: Price vs Consumer Surplus
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for policy in df["policy"].unique():
        subset = df[df["policy"] == policy]
        ax.scatter(subset["mean_price"], subset["consumer_surplus_per_consumer"], label=policy, alpha=0.7, s=100)
    
    ax.set_xlabel("Mean Price", fontsize=11)
    ax.set_ylabel("Consumer Surplus per Consumer", fontsize=11)
    ax.set_title("Price vs Consumer Surplus by Policy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "price_vs_consumer_surplus.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir / 'price_vs_consumer_surplus.png'}")
    plt.close()
    
    print("\n" + "="*60)
    print("All graphs saved to 'figures/' directory")
    print("="*60)

if __name__ == "__main__":
    main()

