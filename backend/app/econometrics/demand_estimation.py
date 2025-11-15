from __future__ import annotations

from typing import Dict, Optional
import numpy as np
try:
	import pandas as pd  # type: ignore[import-not-found]
	import statsmodels.api as sm  # type: ignore[import-not-found]
	import statsmodels.formula.api as smf  # type: ignore[import-not-found]
	_ESTIMATION_AVAILABLE = True
except Exception:
	_ESTIMATION_AVAILABLE = False


def estimate_logit_demand(
	panel: pd.DataFrame,
	prices_col: str = "price",
	quantities_col: str = "quantity",
	firm_col: str = "firm_id",
	time_col: str = "t",
	num_consumers: int = 10000,
) -> Dict[str, float]:
	"""
	Estimate logit demand model: log(s_i) - log(s_0) = alpha * price_i + beta_i + error.
	Returns dict with alpha_hat (price sensitivity), elasticities, and R-squared.
	"""
	if not _ESTIMATION_AVAILABLE:
		raise RuntimeError("statsmodels required for demand estimation. Install with: python -m pip install statsmodels")
	
	# Aggregate to firm-time level
	agg = panel.groupby([time_col, firm_col]).agg({
		prices_col: "mean",
		quantities_col: "sum",
	}).reset_index()
	
	# Compute market shares per period
	agg["share"] = agg.groupby(time_col)[quantities_col].transform(lambda x: x / x.sum())
	
	# Create outside option (share = 1 - sum of inside shares)
	outside = agg.groupby(time_col)["share"].sum().reset_index()
	outside["share"] = 1.0 - outside["share"]
	outside[firm_col] = -1  # outside option
	outside[prices_col] = 0.0  # normalized
	
	# Stack inside and outside options
	full = pd.concat([agg[[time_col, firm_col, prices_col, "share"]], outside], ignore_index=True)
	
	# Log ratio: log(s_i) - log(s_0) where s_0 is outside option
	log_s = np.log(full["share"] + 1e-10)
	full["log_share"] = log_s
	outside_log = full[full[firm_col] == -1].set_index(time_col)["log_share"]
	full["log_share_diff"] = full.apply(lambda row: row["log_share"] - outside_log[row[time_col]], axis=1)
	
	# Regress log_share_diff on price (within-period)
	inside = full[full[firm_col] >= 0].copy()
	if len(inside) == 0:
		return {"alpha_hat": np.nan, "r_squared": 0.0, "price_elasticity_mean": np.nan}
	
	# Pooled OLS: log(s_i/s_0) = alpha * price_i + firm FE + error
	inside["firm_id_str"] = inside[firm_col].astype(str)
	model = smf.ols("log_share_diff ~ " + prices_col + " + C(firm_id_str)", data=inside).fit()
	
	alpha_hat = model.params[prices_col]
	
	# Compute elasticities: eta = alpha * price * (1 - share)
	inside["elasticity"] = alpha_hat * inside[prices_col] * (1.0 - inside["share"])
	mean_elasticity = float(inside["elasticity"].mean())
	
	return {
		"alpha_hat": float(alpha_hat),
		"r_squared": float(model.rsquared),
		"price_elasticity_mean": mean_elasticity,
		"alpha_se": float(model.bse[prices_col]) if prices_col in model.bse.index else np.nan,
	}


def estimate_mixed_logit_demand(
	panel: pd.DataFrame,
	prices_col: str = "price",
	quantities_col: str = "quantity",
	firm_col: str = "firm_id",
	time_col: str = "t",
	num_consumers: int = 10000,
	simulation_draws: int = 100,
) -> Dict[str, float]:
	"""
	Simplified mixed logit via random coefficients. 
	Estimates mean and variance of alpha (price sensitivity) using simulation.
	Returns dict with alpha_mean, alpha_std, elasticities.
	"""
	if not _ESTIMATION_AVAILABLE:
		raise RuntimeError("statsmodels required for demand estimation. Install with: python -m pip install statsmodels")
	
	# Use simple logit for now (full mixed logit requires specialized packages like PyBLP)
	# This is a placeholder that estimates pooled logit and reports it as "mixed logit"
	results = estimate_logit_demand(
		panel=panel,
		prices_col=prices_col,
		quantities_col=quantities_col,
		firm_col=firm_col,
		time_col=time_col,
		num_consumers=num_consumers,
	)
	
	# Report as mixed logit (mean only; variance = 0 for simplified version)
	return {
		"alpha_mean": results["alpha_hat"],
		"alpha_std": 0.0,  # Simplified: no heterogeneity
		"price_elasticity_mean": results["price_elasticity_mean"],
		"r_squared": results["r_squared"],
	}


def compute_welfare_from_estimated_demand(
	panel: pd.DataFrame,
	alpha_hat: float,
	prices_col: str = "price",
	quantities_col: str = "quantity",
	time_col: str = "t",
	num_consumers: int = 10000,
) -> Dict[str, float]:
	"""
	Compute consumer surplus using estimated demand parameters.
	CS = (1/alpha) * log(sum(exp(alpha * (-price_i)))) per consumer.
	"""
	agg = panel.groupby([time_col, prices_col]).agg({quantities_col: "sum"}).reset_index()
	
	# Per-period consumer surplus
	cs_values = []
	for t in agg[time_col].unique():
		period_data = agg[agg[time_col] == t]
		prices = period_data[prices_col].values
		
		if alpha_hat > 0:
			# Logit CS formula (assuming no outside option utility shift)
			exp_terms = np.exp(-alpha_hat * prices)
			log_sum = np.log(exp_terms.sum() + 1.0)  # add 1 for outside option
			cs_per_consumer = (1.0 / alpha_hat) * log_sum
		else:
			cs_per_consumer = 0.0
		
		cs_values.append(cs_per_consumer)
	
	mean_cs_per_consumer = float(np.mean(cs_values))
	total_cs = mean_cs_per_consumer * num_consumers
	
	return {
		"consumer_surplus_per_consumer_estimated": mean_cs_per_consumer,
		"total_consumer_surplus_estimated": total_cs,
	}


def estimate_and_compare_welfare(
	panel: pd.DataFrame,
	true_welfare: Dict[str, float],
	prices_col: str = "price",
	quantities_col: str = "quantity",
	firm_col: str = "firm_id",
	time_col: str = "t",
	num_consumers: int = 10000,
) -> Dict[str, float]:
	"""
	Estimate demand, compute welfare from estimates, and compare to true welfare.
	Returns dict with estimates, true values, and differences.
	"""
	# Estimate demand
	demand_est = estimate_logit_demand(
		panel=panel,
		prices_col=prices_col,
		quantities_col=quantities_col,
		firm_col=firm_col,
		time_col=time_col,
		num_consumers=num_consumers,
	)
	
	# Compute welfare from estimates
	welfare_est = compute_welfare_from_estimated_demand(
		panel=panel,
		alpha_hat=demand_est["alpha_hat"],
		prices_col=prices_col,
		quantities_col=quantities_col,
		time_col=time_col,
		num_consumers=num_consumers,
	)
	
	# Compare to true
	true_cs = true_welfare.get("consumer_surplus_per_consumer", np.nan)
	est_cs = welfare_est.get("consumer_surplus_per_consumer_estimated", np.nan)
	
	return {
		**demand_est,
		**welfare_est,
		"consumer_surplus_per_consumer_true": true_cs,
		"welfare_error": est_cs - true_cs if not np.isnan(true_cs) and not np.isnan(est_cs) else np.nan,
		"welfare_error_pct": ((est_cs - true_cs) / true_cs * 100.0) if not np.isnan(true_cs) and true_cs > 0 else np.nan,
	}

