from memomemo.pareto import ParetoPoint, pareto_frontier


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
