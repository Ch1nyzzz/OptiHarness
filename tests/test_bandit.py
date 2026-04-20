from memomemo.bandit import (
    ArmStats,
    BanditArm,
    BanditState,
    select_ucb_arm,
    update_bandit_state,
)
from memomemo.optimizer import MemoOptimizer, OptimizerConfig
from memomemo.schemas import CandidateResult


def test_select_ucb_arm_prioritizes_unpulled_arms():
    state = BanditState(
        total_pulls=3,
        arms={"mem0|low": ArmStats(pulls=3, mean_reward=0.5)},
    )

    arm = select_ucb_arm(
        state,
        available_arms=[
            BanditArm("mem0", "low"),
            BanditArm("memgpt", "low"),
        ],
    )

    assert arm == BanditArm("memgpt", "low")


def test_select_ucb_arm_uses_reward_and_exploration_bonus():
    state = BanditState(
        total_pulls=12,
        arms={
            "mem0|low": ArmStats(pulls=10, mean_reward=0.1),
            "memgpt|low": ArmStats(pulls=2, mean_reward=0.1),
        },
    )

    arm = select_ucb_arm(
        state,
        available_arms=[
            BanditArm("mem0", "low"),
            BanditArm("memgpt", "low"),
        ],
        exploration_c=0.6,
    )

    assert arm == BanditArm("memgpt", "low")


def test_update_bandit_state_tracks_frontier_hits_and_reward():
    state = BanditState(total_pulls=0, arms={})
    arm = BanditArm("mem0", "medium")

    updated = update_bandit_state(
        state,
        arm,
        reward=0.7,
        entered_frontier=True,
        iteration=3,
    )

    stats = updated.arms["mem0|medium"]
    assert stats.pulls == 1
    assert stats.mean_reward == 0.7
    assert stats.frontier_hits == 1
    assert stats.last_iteration == 3


def test_ucb_reward_applies_small_cost_penalty_when_candidate_enters_frontier(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    parent = _candidate("parent", passrate=0.5, avg_tokens=1000)
    child = _candidate("child", passrate=0.6, avg_tokens=2000)

    reward = optimizer._ucb_round_reward(
        arm=BanditArm("fusion", "high"),
        parent=parent,
        evaluated=[child],
        frontier_ids={"child"},
    )

    assert reward == 1.09


def test_ucb_reward_penalizes_high_cost_when_no_candidate_enters_frontier(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    parent = _candidate("parent", passrate=0.5, avg_tokens=1000)
    child = _candidate("child", passrate=0.5, avg_tokens=2000)

    reward = optimizer._ucb_round_reward(
        arm=BanditArm("fusion", "high"),
        parent=parent,
        evaluated=[child],
        frontier_ids=set(),
    )

    assert reward == -0.01


def test_ucb_reward_ignores_eval_token_regression_when_not_on_frontier(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    parent = _candidate("parent", passrate=0.5, avg_tokens=1000)
    expensive_child = _candidate("expensive_child", passrate=0.55, avg_tokens=3000)
    cheap_child = _candidate("cheap_child", passrate=0.55, avg_tokens=500)

    expensive_reward = optimizer._ucb_round_reward(
        arm=BanditArm("mem0", "low"),
        parent=parent,
        evaluated=[expensive_child],
        frontier_ids=set(),
    )
    cheap_reward = optimizer._ucb_round_reward(
        arm=BanditArm("mem0", "low"),
        parent=parent,
        evaluated=[cheap_child],
        frontier_ids=set(),
    )

    assert expensive_reward == cheap_reward
    assert round(expensive_reward, 3) == 0.05


def test_high_cost_with_one_more_passrate_step_beats_low_cost_no_gain(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    parent = _candidate("parent", passrate=0.5, avg_tokens=1000)
    low_child = _candidate("low_child", passrate=0.5, avg_tokens=500)
    high_child = _candidate("high_child", passrate=0.525, avg_tokens=3000)

    low_reward = optimizer._ucb_round_reward(
        arm=BanditArm("mem0", "low"),
        parent=parent,
        evaluated=[low_child],
        frontier_ids=set(),
    )
    high_reward = optimizer._ucb_round_reward(
        arm=BanditArm("mem0", "high"),
        parent=parent,
        evaluated=[high_child],
        frontier_ids=set(),
    )

    assert low_reward == 0.0
    assert round(high_reward, 3) == 0.015
    assert high_reward > low_reward


def test_ucb_parent_selection_prioritizes_quality_not_tokens(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    cheap_lower_quality = _candidate(
        "cheap_lower_quality",
        passrate=0.525,
        average_score=0.55,
        avg_tokens=500,
        source_family="mem0",
    )
    expensive_higher_quality = _candidate(
        "expensive_higher_quality",
        passrate=0.55,
        average_score=0.57,
        avg_tokens=3000,
        source_family="mem0",
    )

    parent = optimizer._select_parent_for_arm(
        [cheap_lower_quality, expensive_higher_quality],
        BanditArm("mem0", "medium"),
    )

    assert parent.candidate_id == "expensive_higher_quality"


def test_ucb_parent_selection_uses_tokens_not_average_score_as_tiebreaker(tmp_path):
    optimizer = MemoOptimizer(OptimizerConfig(run_id="r", out_dir=tmp_path))
    cheap_lower_average_score = _candidate(
        "cheap_lower_average_score",
        passrate=0.55,
        average_score=0.50,
        avg_tokens=500,
        source_family="mem0",
    )
    expensive_higher_average_score = _candidate(
        "expensive_higher_average_score",
        passrate=0.55,
        average_score=0.90,
        avg_tokens=3000,
        source_family="mem0",
    )

    parent = optimizer._select_parent_for_arm(
        [cheap_lower_average_score, expensive_higher_average_score],
        BanditArm("mem0", "medium"),
    )

    assert parent.candidate_id == "cheap_lower_average_score"


def _candidate(
    candidate_id: str,
    *,
    passrate: float,
    avg_tokens: float,
    average_score: float | None = None,
    source_family: str | None = None,
) -> CandidateResult:
    extra = {"source_family": source_family} if source_family else {}
    return CandidateResult(
        candidate_id=candidate_id,
        scaffold_name=candidate_id,
        passrate=passrate,
        average_score=passrate if average_score is None else average_score,
        token_consuming=int(avg_tokens * 40),
        avg_token_consuming=avg_tokens,
        avg_prompt_tokens=avg_tokens,
        avg_completion_tokens=0,
        count=40,
        config={"extra": extra},
        result_path=f"{candidate_id}.json",
    )
