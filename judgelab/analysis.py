"""Every statistic in JudgeLab, computed from the compact pair file.

Input: one record per (question, model pair, turn) that human experts voted on:

    {"question_id": 81, "turn": 1, "model_a": "...", "model_b": "...", "category": "writing",
     "votes": [{"judge": "expert_3", "winner": "model_a"}, ...],
     "gpt4": "model_b" | "tie" | "tie (inconsistent)" | null,
     "len_a": 1234, "len_b": 987}

``gpt4`` is GPT-4's verdict from MT-Bench's both-orders protocol: the pair was judged
twice with the answers swapped, and a verdict that flipped is recorded as
"tie (inconsistent)".
"""
from __future__ import annotations

from itertools import combinations

from .stats import agreement, estimate, kappa, majority

LENGTH_GAP = 50  # characters; below this, "longer" is not meaningful


def norm(label: str) -> str:
    return "tie" if label.startswith("tie") else label


def _pairs(rows, a="h", b="g"):
    return [r[a] for r in rows], [r[b] for r in rows]


def _kappa(rows, a="h", b="g"):
    return kappa(*_pairs(rows, a, b))


def _agree(rows, a="h", b="g"):
    return agreement(*_pairs(rows, a, b))


def joined_rows(pairs: list[dict]) -> list[dict]:
    """Pairs judged by both GPT-4 and humans, with the human majority as ``h``."""
    out = []
    for p in pairs:
        if p.get("gpt4") is None:
            continue
        out.append({**p, "h": majority([norm(v["winner"]) for v in p["votes"]]),
                    "g": norm(p["gpt4"]), "g_raw": p["gpt4"]})
    return out


def headline(rows: list[dict]) -> dict:
    decisive = [r for r in rows if r["h"] != "tie" and r["g"] != "tie"]
    return {
        "n_joined_pairs": len(rows),
        "agreement_all": estimate(rows, _agree),
        "kappa_all": estimate(rows, _kappa),
        "n_nontie": len(decisive),
        "agreement_nontie": estimate(decisive, _agree),
        "kappa_nontie": estimate(decisive, _kappa),
    }


def human_baseline(pairs: list[dict]) -> dict:
    """Is GPT-4 further from a human than humans are from each other?

    On every pair with at least two human votes and a GPT-4 verdict, compare
    (a) each human vote against every other human vote on the same pair, and
    (b) each human vote against GPT-4. Same pairs, same unit (a single judge), so the
    two numbers are directly comparable.
    """
    hh, gh = [], []
    for p in pairs:
        if p.get("gpt4") is None or len(p["votes"]) < 2:
            continue
        labels = [norm(v["winner"]) for v in p["votes"]]
        for x, y in combinations(labels, 2):
            hh.append({"question_id": p["question_id"], "a": x, "b": y})
        for x in labels:
            gh.append({"question_id": p["question_id"], "a": x, "b": norm(p["gpt4"])})
    n_pairs = sum(1 for p in pairs if p.get("gpt4") is not None and len(p["votes"]) >= 2)
    return {
        "n_pairs": n_pairs,
        "human_vs_human": {"agreement": estimate(hh, lambda r: _agree(r, "a", "b")),
                           "kappa": estimate(hh, lambda r: _kappa(r, "a", "b"))},
        "gpt4_vs_human": {"agreement": estimate(gh, lambda r: _agree(r, "a", "b")),
                          "kappa": estimate(gh, lambda r: _kappa(r, "a", "b"))},
    }


def position(rows: list[dict], all_gpt4_labels: list[dict]) -> dict:
    flipped = [{"question_id": r["question_id"], "x": r["gpt4"] == "tie (inconsistent)"}
               for r in all_gpt4_labels]
    rate = estimate(flipped, lambda rs: sum(r["x"] for r in rs) / len(rs))
    return {"n_judgments": len(flipped), "inconsistent_count": sum(r["x"] for r in flipped),
            "inconsistent_rate": rate}


def selective(rows: list[dict]) -> dict:
    """Abstain whenever the two orderings disagree: what does the judge buy with it?"""
    consistent = [r for r in rows if r["g_raw"] != "tie (inconsistent)"]
    decisive = [r for r in consistent if r["h"] != "tie" and r["g"] != "tie"]
    return {
        "coverage": round(len(consistent) / len(rows), 4),
        "n_consistent": len(consistent),
        "agreement": estimate(consistent, _agree),
        "kappa": estimate(consistent, _kappa),
        "kappa_decisive": estimate(decisive, _kappa),
        "n_decisive": len(decisive),
    }


def verbosity(rows: list[dict]) -> dict:
    gap = [r for r in rows if abs(r["len_a"] - r["len_b"]) > LENGTH_GAP]

    def longer(r):
        return "model_a" if r["len_a"] > r["len_b"] else "model_b"

    def pref(who):
        sub = [r for r in gap if r[who] != "tie"]
        return estimate(sub, lambda rs: sum(r[who] == longer(r) for r in rs) / len(rs))

    # Where both picked a winner but different ones, who took the longer answer?
    split = [r for r in gap if "tie" not in (r["h"], r["g"]) and r["h"] != r["g"]]
    return {
        "length_gap_chars": LENGTH_GAP,
        "human_prefers_longer": pref("h"),
        "gpt4_prefers_longer": pref("g"),
        "disagreements_gpt4_took_longer": estimate(
            split, lambda rs: sum(r["g"] == longer(r) for r in rs) / len(rs)),
    }


def per_category(rows: list[dict]) -> dict:
    out = {}
    for c in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == c]
        out[c] = {"n": len(sub), "agreement": estimate(sub, _agree, n_boot=1000),
                  "kappa": estimate(sub, _kappa, n_boot=1000)}
    return out


def category_gap(rows: list[dict], weak: str, strong: str) -> dict:
    """Is the gap between two categories real, or noise? Bootstrap the difference."""
    sub = [r for r in rows if r["category"] in (weak, strong)]

    def diff(rs):
        s = [r for r in rs if r["category"] == strong]
        w = [r for r in rs if r["category"] == weak]
        return (_kappa(s) if s else 0.0) - (_kappa(w) if w else 0.0)

    return {"weak": weak, "strong": strong, "kappa_difference": estimate(sub, diff, n_boot=1000)}


def compute(pairs: list[dict], gpt4_judgments: list[dict]) -> dict:
    rows = joined_rows(pairs)
    cats = per_category(rows)
    ranked = sorted(cats, key=lambda c: cats[c]["kappa"]["value"])
    return {
        "source": "lmsys/mt_bench_human_judgments (Hugging Face), splits human + gpt4_pair",
        "ci_method": "95% percentile bootstrap, resampling whole questions (clustered)",
        "n_human_votes": sum(len(p["votes"]) for p in pairs),
        "n_gpt4_judgments": len(gpt4_judgments),
        **headline(rows),
        "human_baseline": human_baseline(pairs),
        "position": position(rows, gpt4_judgments),
        "selective": selective(rows),
        "verbosity": verbosity(rows),
        "per_category": cats,
        "category_gap": category_gap(rows, ranked[0], ranked[-1]),
    }
