from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Regulation(BaseModel):
	price_floor: Optional[float] = Field(default=None, ge=0.0)
	price_ceiling: Optional[float] = Field(default=None, ge=0.0)
	audit_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
	cooldown_steps: Optional[int] = Field(default=None, ge=0)


class DemandParams(BaseModel):
	price_sensitivity: float = Field(..., gt=0.0, description="Alpha in logit demand; higher means more elastic.")
	mean_quality: float = Field(0.0, description="Average quality/taste shifter.")


class CostProcess(BaseModel):
	initial_cost: float = Field(1.0, ge=0.0)
	sigma: float = Field(0.1, ge=0.0, description="Std dev of cost innovations per period.")


class Scenario(BaseModel):
	id: Optional[str] = None
	name: str
	num_firms: int = Field(..., gt=0)
	num_consumers: int = Field(..., gt=0)
	time_periods: int = Field(..., gt=0)
	demand: DemandParams
	cost_process: CostProcess
	regulation: Optional[Regulation] = None
	# Pricing/learning policy
	policy: str = Field("myopic", description="Pricing policy: 'myopic' | 'epsilon_bandit' | 'q_learning' | 'actor_critic'")
	policy_params: Dict[str, float] = Field(default_factory=dict, description="Policy tuning: epsilon, markup_min/max, num_actions, learning_rate, discount, num_state_bins (for Q-learning/AC)")


class ScenarioCreate(BaseModel):
	name: str
	num_firms: int = Field(..., gt=0)
	num_consumers: int = Field(..., gt=0)
	time_periods: int = Field(..., gt=0)
	demand: DemandParams
	cost_process: CostProcess
	regulation: Optional[Regulation] = None
	policy: str = Field("myopic", description="Pricing policy: 'myopic' | 'epsilon_bandit' | 'q_learning' | 'actor_critic'")
	policy_params: Dict[str, float] = Field(default_factory=dict, description="Policy tuning: epsilon, markup_min/max, num_actions, learning_rate, discount, num_state_bins (for Q-learning/AC)")


class RunConfig(BaseModel):
	seed: int = 42
	log_interval: int = Field(10, gt=0)


class RunCreate(BaseModel):
	scenario_id: Optional[str] = None
	scenario: Optional[ScenarioCreate] = None
	config: RunConfig = RunConfig()


class RunStatus(BaseModel):
	run_id: str
	status: str = Field(description="queued|running|succeeded|failed")
	message: Optional[str] = None


class RunSummary(BaseModel):
	run_id: str
	mean_price: float
	mean_profit_per_firm: float
	consumer_surplus_per_consumer: float
	total_welfare: float

