"""Agreement statistics with honest uncertainty. Pure Python, no dependencies.

MT-Bench has 80 questions, and each contributes many answer pairs. Pairs from the same
question are not independent (one confusing question drags every pair on it), so the
confidence intervals here resample whole questions, not individual pairs. A naive
pair-level bootstrap would report intervals that are too narrow.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence


def agreement(a: Sequence[str], b: Sequence[str]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else 0.0


def kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for any number of categories (here: model_a / model_b / tie)."""
    n = len(a)
    if n == 0:
        return 0.0
    po = agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def cluster_bootstrap(rows: Sequence[dict], stat: Callable[[list[dict]], float],
                      cluster: str = "question_id", n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for ``stat(rows)``, resampling whole clusters with replacement."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[cluster], []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        draws.append(stat(sample))
    draws.sort()
    return draws[int(alpha / 2 * n_boot)], draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def estimate(rows: Sequence[dict], stat: Callable[[list[dict]], float], **kw) -> dict:
    """Point estimate plus clustered 95% CI, rounded for publication."""
    lo, hi = cluster_bootstrap(rows, stat, **kw)
    return {"value": round(stat(list(rows)), 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(rows)}


def majority(votes: Sequence[str]) -> str:
    """Plurality label; a tie between labels counts as a 'tie' verdict."""
    counts = Counter(votes)
    best = max(counts.values())
    winners = [v for v, c in counts.items() if c == best]
    return winners[0] if len(winners) == 1 else "tie"
