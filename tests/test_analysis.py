import json
import pathlib

import pytest

from judgelab.analysis import compute, human_baseline, joined_rows, selective, verbosity
from judgelab.data import load_compact

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pair(qid, votes, gpt4, la=100, lb=300, cat="writing"):
    return {"question_id": qid, "turn": 1, "model_a": "m1", "model_b": "m2", "category": cat,
            "votes": [{"judge": f"j{i}", "winner": w} for i, w in enumerate(votes)],
            "gpt4": gpt4, "len_a": la, "len_b": lb}


def test_joined_rows_use_human_majority_and_normalise_inconsistent():
    rows = joined_rows([pair(1, ["model_a", "model_a", "model_b"], "tie (inconsistent)"),
                        pair(2, ["model_b"], None)])
    assert len(rows) == 1
    assert rows[0]["h"] == "model_a" and rows[0]["g"] == "tie"


def test_human_baseline_compares_like_with_like():
    pairs = [pair(q, ["model_a", "model_b"], "model_a") for q in range(10)]
    hb = human_baseline(pairs)
    assert hb["n_pairs"] == 10
    assert hb["human_vs_human"]["agreement"]["value"] == 0.0  # the two humans always split
    assert hb["gpt4_vs_human"]["agreement"]["value"] == 0.5   # GPT-4 matches one of them


def test_selective_drops_only_inconsistent_verdicts():
    rows = joined_rows([pair(1, ["model_a"], "model_a"), pair(2, ["model_b"], "tie (inconsistent)"),
                        pair(3, ["model_b"], "model_b"), pair(4, ["model_a"], "tie")])
    s = selective(rows)
    assert s["n_consistent"] == 3 and s["coverage"] == 0.75


def test_verbosity_direction_in_disagreements():
    # b is longer; humans pick a, GPT-4 picks b -> GPT-4 took the longer answer every time
    rows = joined_rows([pair(q, ["model_a"], "model_b") for q in range(6)])
    assert verbosity(rows)["disagreements_gpt4_took_longer"]["value"] == 1.0


@pytest.fixture(scope="module")
def real():
    return compute(**load_compact())


def test_real_data_counts(real):
    assert real["n_human_votes"] == 3355
    assert real["n_gpt4_judgments"] == 2400
    assert real["n_joined_pairs"] == 1232
    assert real["position"]["inconsistent_count"] == 380


def test_real_data_headline_values(real):
    assert real["kappa_all"]["value"] == pytest.approx(0.41, abs=1e-3)
    assert real["kappa_nontie"]["value"] == pytest.approx(0.4614, abs=1e-3)
    for key in ("kappa_all", "kappa_nontie", "agreement_all"):
        lo, hi = real[key]["ci"]
        assert lo < real[key]["value"] < hi


def test_committed_stats_are_current(real):
    committed = json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(real))
