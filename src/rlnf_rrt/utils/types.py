from __future__ import annotations

from dataclasses import dataclass

from rlnf_rrt.models.flow import Flow
from rlnf_rrt.models.subgoal_policy import SubGoalPolicy
from rlnf_rrt.models.recursive_subgoal_policy import RecursiveSubgoalPolicy

@dataclass
class RolloutModels:
    split_agent: RecursiveSubgoalPolicy
    flow_model: Flow
    kl_anchor_model: SubGoalPolicy | None

@dataclass
class RolloutConfig:
    max_depth: int
    num_samples: int
    pts_per_seg: int
    rrt_cfg: dict
    total_budget: int
    min_iter: int
    budget_scale: float
    clearance_penalty_coef: float
    img_size: int
    # Midpoint reward weights
    mid_w_clearance: float = 0.3
    mid_w_balance: float = 0.3
    mid_w_child: float = 0.4

@dataclass
class DynamicRolloutParams:
    baseline_reward: float
    baseline_is_goal: bool
    force_split_eps: float
    complexity_weight: float
    split_explore_factor: float

