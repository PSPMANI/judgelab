"""Build and load the compact dataset that every statistic is computed from.

The raw Hugging Face splits are 27 MB (they carry full conversations). Everything the
analysis needs fits in a ~40 KB gzip, which is committed, so CI can regenerate every
number without touching the network.
"""
from __future__ import annotations

import gzip
import json
import pathlib

CATEGORIES = [
    (81, 90, "writing"), (91, 100, "roleplay"), (101, 110, "reasoning"),
    (111, 120, "math"), (121, 130, "coding"), (131, 140, "extraction"),
    (141, 150, "stem"), (151, 160, "humanities"),
]

COMPACT = pathlib.Path(__file__).resolve().parent.parent / "data" / "mtbench_compact.json.gz"


def category(qid: int) -> str:
    for lo, hi, name in CATEGORIES:
        if lo <= qid <= hi:
            return name
    return "other"


def pair_key(r: dict) -> tuple:
    return (r["question_id"], r["model_a"], r["model_b"], r["turn"])


def answer_text(conv: list[dict]) -> str:
    return " ".join(m.get("content", "") for m in conv if m.get("role") == "assistant")


def build_compact(human_rows: list[dict], gpt4_rows: list[dict]) -> dict:
    gpt4 = {pair_key(r): r["winner"] for r in gpt4_rows}
    pairs: dict = {}
    for r in human_rows:
        k = pair_key(r)
        if k not in pairs:
            qid, ma, mb, turn = k
            pairs[k] = {"question_id": qid, "turn": turn, "model_a": ma, "model_b": mb,
                        "category": category(qid), "votes": [], "gpt4": gpt4.get(k),
                        "len_a": len(answer_text(r["conversation_a"])),
                        "len_b": len(answer_text(r["conversation_b"]))}
        pairs[k]["votes"].append({"judge": r["judge"], "winner": r["winner"]})
    return {
        "pairs": list(pairs.values()),
        "gpt4_judgments": [{"question_id": r["question_id"], "turn": r["turn"],
                            "model_a": r["model_a"], "model_b": r["model_b"],
                            "gpt4": r["winner"]} for r in gpt4_rows],
    }


def save_compact(data: dict, path: pathlib.Path = COMPACT) -> None:
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # mtime=0 keeps the gzip byte-identical across runs, so git sees no spurious change
    path.write_bytes(gzip.compress(raw, mtime=0))


def load_compact(path: pathlib.Path = COMPACT) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
