"""Download and process the real MT-Bench judge-reliability data.

    python prepare_data.py

Pulls two public splits of lmsys/mt_bench_human_judgments from the Hugging Face
datasets server (no API key, no auth):

  - "human":     3.3k pairwise votes by human experts
  - "gpt4_pair": GPT-4's pairwise judgments on the same answer pairs

then writes:

  data/mtbench_compact.json.gz  - every vote and verdict, without the long answer
                                  texts (~30 KB, committed; all statistics use it)
  data/gallery.json             - a browsable sample of pairs with full answer text

Then `python analyze.py` computes every number from the compact file, offline.
"""
import json
import pathlib
import time
import urllib.parse
import urllib.request

from judgelab.data import answer_text, build_compact, pair_key, save_compact

BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "lmsys/mt_bench_human_judgments"
OUT = pathlib.Path(__file__).parent / "data"

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


def build_gallery(human_rows, compact):
    """A browsable sample with full answer text: every disagreement type first."""
    from judgelab.analysis import joined_rows

    text = {}
    for r in human_rows:
        k = pair_key(r)
        if k not in text:
            q = next((m["content"] for m in r["conversation_a"] if m["role"] == "user"), "")
            text[k] = (q[:400], answer_text(r["conversation_a"])[:900], answer_text(r["conversation_b"])[:900])
    rows = joined_rows(compact["pairs"])
    out = []
    for j in rows:
        q, a, b = text[pair_key(j)]
        out.append({"question_id": j["question_id"], "model_a": j["model_a"], "model_b": j["model_b"],
                    "turn": j["turn"], "category": j["category"], "human": j["h"], "gpt4": j["g"],
                    "gpt4_raw": j["gpt4"], "n_human_votes": len(j["votes"]),
                    "len_a": j["len_a"], "len_b": j["len_b"], "question": q, "answer_a": a, "answer_b": b})
    disagreements = sorted((g for g in out if g["human"] != g["gpt4"]),
                           key=lambda g: (g["category"], g["question_id"]))
    agreements = [g for g in out if g["human"] == g["gpt4"]]
    return disagreements[:40] + agreements[:20]


def main():
    OUT.mkdir(exist_ok=True)
    print("Downloading public MT-Bench judgment data (no key needed)...")
    human_rows = fetch_split("human")
    gpt4_rows = fetch_split("gpt4_pair")
    compact = build_compact(human_rows, gpt4_rows)
    save_compact(compact)
    gallery = build_gallery(human_rows, compact)
    (OUT / "gallery.json").write_text(json.dumps(gallery, indent=2), encoding="utf-8")
    print(f"Wrote data/mtbench_compact.json.gz ({len(compact['pairs'])} pairs) and "
          f"data/gallery.json ({len(gallery)} pairs). Now run: python analyze.py")


if __name__ == "__main__":
    main()
