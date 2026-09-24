import random

import pytest

from judgelab.stats import agreement, cluster_bootstrap, estimate, kappa, majority


def test_kappa_textbook_binary():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert kappa(a, b) == pytest.approx(0.4)


def test_kappa_three_class_matches_hand_calculation():
    a = ["model_a", "model_a", "model_b", "tie", "model_b", "tie"]
    b = ["model_a", "model_b", "model_b", "tie", "model_a", "model_b"]
    po = 3 / 6
    pe = (2 * 2 + 2 * 3 + 2 * 1) / 36
    assert kappa(a, b) == pytest.approx((po - pe) / (1 - pe))


def test_kappa_perfect_and_constant():
    assert kappa(["x", "y"], ["x", "y"]) == 1.0
    assert kappa(["x", "x"], ["x", "x"]) == 1.0


def test_agreement_requires_equal_lengths():
    with pytest.raises(ValueError):
        agreement(["a"], ["a", "b"])


def test_majority_breaks_ties_as_tie():
    assert majority(["model_a", "model_a", "model_b"]) == "model_a"
    assert majority(["model_a", "model_b"]) == "tie"


def rate(rows):
    return sum(r["x"] for r in rows) / len(rows)


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = random.Random(1)
    rows = [{"question_id": i // 5, "x": rng.random() < 0.3} for i in range(200)]
    e = estimate(rows, rate)
    assert e["ci"][0] <= e["value"] <= e["ci"][1]
    assert estimate(rows, rate) == e


def test_clustered_interval_is_wider_when_clusters_are_correlated():
    # 20 questions x 10 pairs; every pair on a question shares its outcome.
    rng = random.Random(2)
    rows = []
    for q in range(20):
        outcome = rng.random() < 0.5
        rows += [{"question_id": q, "pair": q * 10 + k, "x": outcome} for k in range(10)]
    lo_c, hi_c = cluster_bootstrap(rows, rate, cluster="question_id")
    lo_n, hi_n = cluster_bootstrap(rows, rate, cluster="pair")
    assert (hi_c - lo_c) > 1.8 * (hi_n - lo_n)
