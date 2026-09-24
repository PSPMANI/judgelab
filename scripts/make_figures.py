"""Regenerate the README figures from data/stats.json: python scripts/make_figures.py"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))

BG, FG, MUTED, GRID = "#0E1117", "#E6EAF1", "#94A3B8", "#2A3140"
PURPLE, BLUE, RED, GREEN = "#8B5CF6", "#3B82F6", "#EF4444", "#22C55E"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": GRID, "font.size": 10})


def forest(ax, rows, xlabel):
    for i, (_label, e, color) in enumerate(rows):
        lo, hi = e["ci"]
        ax.plot([lo, hi], [i, i], color=color, lw=3, alpha=0.45, solid_capstyle="round")
        ax.plot(e["value"], i, "o", color=color, ms=8)
        ax.text(1.02, i, f"{e['value']:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=9,
                color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color=GRID, lw=1)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def baseline_and_categories(out):
    hb = S["human_baseline"]
    top = [("Human vs another human", hb["human_vs_human"]["kappa"], GREEN),
           ("GPT-4 vs a human", hb["gpt4_vs_human"]["kappa"], PURPLE)]
    cats = sorted(S["per_category"].items(), key=lambda kv: kv[1]["kappa"]["value"])
    bottom = [(c, v["kappa"], RED if v["kappa"]["value"] < 0.3 else BLUE) for c, v in cats]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), gridspec_kw={"height_ratios": [2, 8]})
    forest(a1, top, "")
    a1.set_title(f"Same {hb['n_pairs']} multi-vote pairs: GPT-4 is about as close to a human "
                 "as humans are to each other", fontsize=10.5, loc="left", pad=10)
    forest(a2, bottom, "Cohen's kappa vs human experts, 95% CI (bootstrap, clustered by question)")
    a2.set_title("By category: wide intervals, and writing vs math overlaps", fontsize=10.5,
                 loc="left", pad=10)
    for a in (a1, a2):
        a.set_xlim(-0.1, 0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def verbosity(out):
    v = S["verbosity"]
    rows = [("Humans pick the longer answer", v["human_prefers_longer"], BLUE),
            ("GPT-4 picks the longer answer", v["gpt4_prefers_longer"], PURPLE),
            ("When they disagree, GPT-4 took the longer one", v["disagreements_gpt4_took_longer"], RED)]
    fig, ax = plt.subplots(figsize=(9, 2.4))
    forest(ax, rows, f"share of decisive verdicts on pairs whose lengths differ by >{v['length_gap_chars']} chars")
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlim(0.4, 0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    baseline_and_categories(docs / "reliability.png")
    verbosity(docs / "verbosity.png")
    print("Wrote docs/reliability.png and docs/verbosity.png")
