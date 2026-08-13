"""Download and process the real MT-Bench judge-reliability data.

    python prepare_data.py

Pulls two public splits of lmsys/mt_bench_human_judgments from the Hugging Face
datasets server (no API key, no auth):

  - "human":     3.3k pairwise votes by human experts
  - "gpt4_pair": GPT-4's pairwise judgments on the same answer pairs

then joins them on (question_id, model_a, model_b, turn) and computes every
statistic the app shows: agreement, Cohen's kappa, position-consistency,
verbosity bias, and per-category reliability. Writes:

  data/stats.json    - all aggregate numbers (small, committed)
  data/gallery.json  - a browsable sample of pairs incl. every disagreement type

Reproducible: rerunning regenerates both files from the public source.
"""
import json
import pathlib
import time
import urllib.parse
import urllib.request

BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "lmsys/mt_bench_human_judgments"
OUT = pathlib.Path(__file__).parent / "data"

CATEGORIES = [
    (81, 90, "writing"), (91, 100, "roleplay"), (101, 110, "reasoning"),
    (111, 120, "math"), (121, 130, "coding"), (131, 140, "extraction"),
    (141, 150, "stem"), (151, 160, "humanities"),
]


def category(qid):
    for lo, hi, name in CATEGORIES:
        if lo <= qid <= hi:
            return name
    return "other"


def fetch_split(split):
    """Download one split with local caching, resume, and 429 backoff."""
    cache = OUT / f"raw_{split}.json"
    rows = []
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  {split}: resuming from cache ({len(rows)} rows)")
    offset = len(rows)
    while True:
        q = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": split,
            "offset": offset, "length": 100,
        })
        req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": "judgelab-prep"})
        payload = None
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    payload = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(60, 10 * (attempt + 1))
                    print(f"\n  {split}: rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == 7:
                    raise
                else:
                    time.sleep(3 * (attempt + 1))
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(3 * (attempt + 1))
        if payload is None:
            raise RuntimeError(f"could not fetch {split} at offset {offset}")
        batch = [x["row"] for x in payload.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        cache.write_text(json.dumps(rows), encoding="utf-8")
        print(f"  {split}: {offset} rows", end="\r")
        time.sleep(1.2)  # be polite to the free API
        if len(batch) < 100:
            break
    print(f"  {split}: {len(rows)} rows total")
    return rows


def answer_text(conv):
    return " ".join(m.get("content", "") for m in conv if m.get("role") == "assistant")


def key(r):
    return (r["question_id"], r["model_a"], r["model_b"], r["turn"])


def majority(votes):
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    winners = [v for v, c in counts.items() if c == best]
    return winners[0] if len(winners) == 1 else "tie"


def multiclass_kappa(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def main():
    OUT.mkdir(exist_ok=True)
    print("Downloading public MT-Bench judgment data (no key needed)...")
    human_rows = fetch_split("human")
    gpt4_rows = fetch_split("gpt4_pair")

    # aggregate human votes per pair
    human = {}
    for r in human_rows:
        human.setdefault(key(r), {"votes": [], "row": r})
        human[key(r)]["votes"].append(r["winner"])

    # gpt4 verdict per pair (also capture raw winner strings for consistency stats)
    gpt4 = {}
    for r in gpt4_rows:
        gpt4.setdefault(key(r), []).append(r["winner"])

    raw_gpt4_labels = [w for ws in gpt4.values() for w in ws]
    inconsistent = sum(1 for w in raw_gpt4_labels if "inconsistent" in w)

    def norm(w):
        return "tie" if w.startswith("tie") else w

    joined = []
    for k, h in human.items():
        if k not in gpt4:
            continue
        qid, ma, mb, turn = k
        r = h["row"]
        hlab = majority([norm(v) for v in h["votes"]])
        glab = majority([norm(v) for v in gpt4[k]])
        la = len(answer_text(r["conversation_a"]))
        lb = len(answer_text(r["conversation_b"]))
        joined.append({
            "question_id": qid, "model_a": ma, "model_b": mb, "turn": turn,
            "category": category(qid),
            "human": hlab, "gpt4": glab,
            "n_human_votes": len(h["votes"]),
            "len_a": la, "len_b": lb,
            "question": next((m["content"] for m in r["conversation_a"] if m["role"] == "user"), "")[:400],
            "answer_a": answer_text(r["conversation_a"])[:900],
            "answer_b": answer_text(r["conversation_b"])[:900],
        })

    print(f"joined pairs (human + gpt4 on identical pair): {len(joined)}")

    # ---- stats ----
    hl = [j["human"] for j in joined]
    gl = [j["gpt4"] for j in joined]
    n = len(joined)
    agree = sum(1 for a, b in zip(hl, gl) if a == b)
    kappa_all = multiclass_kappa(hl, gl)

    nontie = [j for j in joined if j["human"] != "tie" and j["gpt4"] != "tie"]
    agree_nt = sum(1 for j in nontie if j["human"] == j["gpt4"])
    kappa_nt = multiclass_kappa([j["human"] for j in nontie], [j["gpt4"] for j in nontie])

    # verbosity: among decisive verdicts on pairs with a meaningful length gap
    def longer_pref(rows_, who):
        rows_ = [j for j in rows_ if abs(j["len_a"] - j["len_b"]) > 50 and j[who] != "tie"]
        if not rows_:
            return 0.0, 0
        pref = sum(1 for j in rows_
                   if (j[who] == "model_a") == (j["len_a"] > j["len_b"]))
        return pref / len(rows_), len(rows_)

    vh, nh = longer_pref(joined, "human")
    vg, ng = longer_pref(joined, "gpt4")

    per_cat = {}
    for c in sorted({j["category"] for j in joined}):
        sub = [j for j in joined if j["category"] == c]
        per_cat[c] = {
            "n": len(sub),
            "agreement": round(sum(1 for j in sub if j["human"] == j["gpt4"]) / len(sub), 4),
            "kappa": round(multiclass_kappa([j["human"] for j in sub], [j["gpt4"] for j in sub]), 4),
        }

    stats = {
        "source": "lmsys/mt_bench_human_judgments (Hugging Face), splits human + gpt4_pair",
        "n_human_votes": len(human_rows),
        "n_gpt4_judgments": len(gpt4_rows),
        "n_joined_pairs": n,
        "agreement_all": round(agree / n, 4),
        "kappa_all": round(kappa_all, 4),
        "n_nontie": len(nontie),
        "agreement_nontie": round(agree_nt / len(nontie), 4),
        "kappa_nontie": round(kappa_nt, 4),
        "gpt4_inconsistent_count": inconsistent,
        "gpt4_inconsistent_rate": round(inconsistent / max(1, len(raw_gpt4_labels)), 4),
        "verbosity_pref_human": round(vh, 4), "verbosity_n_human": nh,
        "verbosity_pref_gpt4": round(vg, 4), "verbosity_n_gpt4": ng,
        "per_category": per_cat,
    }

    # ---- gallery: every kind of case, disagreements first ----
    disagreements = [j for j in joined if j["human"] != j["gpt4"]]
    agreements = [j for j in joined if j["human"] == j["gpt4"]]
    disagreements.sort(key=lambda j: (j["category"], j["question_id"]))
    gallery = disagreements[:40] + agreements[:20]

    OUT.mkdir(exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUT / "gallery.json").write_text(json.dumps(gallery, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "per_category"}, indent=2))
    print("per-category:", json.dumps(stats["per_category"], indent=2))
    print(f"gallery: {len(gallery)} pairs ({len(disagreements[:40])} disagreements)")
    print("Wrote data/stats.json and data/gallery.json")


if __name__ == "__main__":
    main()
