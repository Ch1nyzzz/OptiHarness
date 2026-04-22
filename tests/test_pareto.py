from memomemo.pareto import ParetoPoint, pareto_frontier
from memomemo.optimizer import OptimizerConfig


def point(name, passrate, tokens):
    return ParetoPoint(
        candidate_id=name,
        scaffold_name=name.split("_")[0],
        passrate=passrate,
        token_consuming=tokens,
        avg_token_consuming=tokens,
        average_score=passrate,
        result_path=f"{name}.json",
        config={},
    )


def test_pareto_keeps_tradeoffs_and_drops_dominated():
    frontier = pareto_frontier(
        [
            point("weak_expensive", 0.4, 200),
            point("strong_expensive", 0.8, 300),
            point("strong_cheap", 0.8, 100),
            point("cheap_weak", 0.5, 80),
        ]
    )
    assert [item.candidate_id for item in frontier] == ["strong_cheap", "cheap_weak"]


def test_pareto_quality_threshold_drops_much_weaker_cheap_points():
    frontier = pareto_frontier(
        [
            point("strong_expensive", 0.8, 300),
            point("cheap_much_weaker", 0.74, 50),
        ],
        quality_gap_threshold=0.03,
    )
    assert [item.candidate_id for item in frontier] == ["strong_expensive"]


def test_pareto_quality_threshold_keeps_near_quality_token_tradeoff():
    frontier = pareto_frontier(
        [
            point("strong_expensive", 0.8, 300),
            point("cheap_near_quality", 0.78, 50),
        ],
        quality_gap_threshold=0.03,
    )
    assert [item.candidate_id for item in frontier] == [
        "strong_expensive",
        "cheap_near_quality",
    ]


def test_optimizer_default_pareto_quality_threshold_is_full_train_friendly(tmp_path):
    config = OptimizerConfig(run_id="r", out_dir=tmp_path)

    assert config.pareto_quality_threshold == 0.125
