"""JudgeLab - how much can you trust an LLM as a judge?

Measures GPT-4-as-a-judge against 3.3k real human expert votes from the MT-Bench study
(lmsys/mt_bench_human_judgments): agreement and Cohen's kappa with question-clustered
confidence intervals, a human-vs-human baseline, position consistency, verbosity bias,
per-category reliability, and a gallery of the actual cases where the judge and the
humans disagreed.

All data is real, public, and precomputed: no API key, zero cost. `python analyze.py`
regenerates every number from the committed data.
"""
import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="JudgeLab - LLM-as-a-Judge Reliability", layout="wide")

DATA = pathlib.Path(__file__).parent / "data"

GREEN = "#22C55E"
RED = "#EF4444"
ACCENT = "#8B5CF6"
BLUE = "#3B82F6"
MUTED = "#94A3B8"

CSS = """
<style>
.muted {color:#94A3B8;font-size:0.85rem;}
.verdict {display:inline-block;padding:3px 12px;border-radius:12px;color:white;font-weight:700;font-size:0.85rem;}
.qbox {padding:10px 14px;background:rgba(139,92,246,0.08);border-left:4px solid #8B5CF6;border-radius:6px;margin-bottom:10px;}
.abox {padding:10px 14px;background:rgba(148,163,184,0.07);border-radius:6px;font-size:0.9rem;}
.finding {padding:12px 16px;background:rgba(34,197,94,0.07);border-left:4px solid #22C55E;border-radius:6px;margin:8px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    stats = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
    gallery = json.loads((DATA / "gallery.json").read_text(encoding="utf-8"))
    return stats, gallery


def pill(label):
    color = {"model_a": BLUE, "model_b": ACCENT, "tie": "#64748B"}.get(label, "#64748B")
    text = {"model_a": "A wins", "model_b": "B wins", "tie": "Tie"}.get(label, label)
    return f"<span class='verdict' style='background:{color}'>{text}</span>"


def ci(e, pct=False):
    lo, hi = e["ci"]
    return f"{lo:.1%} to {hi:.1%}" if pct else f"{lo:.2f} to {hi:.2f}"


def forest(rows, x_title, domain):
    """Point estimates with 95% CI bars. rows: list of dicts with label/value/lo/hi/color."""
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort=None, title=None,
                                        axis=alt.Axis(labelLimit=320)))
    bars = base.mark_rule(strokeWidth=4, opacity=0.5).encode(
        x=alt.X("lo:Q", title=x_title, scale=alt.Scale(domain=domain)), x2="hi:Q",
        color=alt.Color("color:N", scale=None))
    dots = base.mark_circle(size=140, opacity=1).encode(
        x="value:Q", color=alt.Color("color:N", scale=None),
        tooltip=["label", alt.Tooltip("value:Q", format=".3f"),
                 alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")])
    return (bars + dots).properties(height=48 * len(rows) + 30)


def row(label, e, color):
    return {"label": label, "value": e["value"], "lo": e["ci"][0], "hi": e["ci"][1], "color": color}


stats, gallery = load()
hb = stats["human_baseline"]

st.sidebar.markdown("## JudgeLab")
st.sidebar.caption("LLM-as-a-Judge Reliability Lab · v2")
st.sidebar.markdown(
    f"Built on **{stats['n_human_votes']:,} real human expert votes** and "
    f"**{stats['n_gpt4_judgments']:,} real GPT-4 judgments** from the MT-Bench "
    "study (lmsys/mt_bench_human_judgments)."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Every interval is a 95% bootstrap CI that **resamples whole questions**, because "
    "pairs on the same question are not independent. `python analyze.py` regenerates "
    "every number offline; CI fails if any published number drifts."
)
st.sidebar.markdown(
    "<span class='muted'>Companion project to TrajLens: TrajLens grades agents "
    "with deterministic rubrics; JudgeLab measures the automated judge itself.</span>",
    unsafe_allow_html=True,
)

tab_overview, tab_bias, tab_gallery, tab_about = st.tabs(
    ["Reliability report", "Bias analysis", "Disagreement gallery", "Methodology"]
)

# ---- Overview ------------------------------------------------------------
with tab_overview:
    st.subheader("Can you trust GPT-4 as a judge? Measured, with error bars.")
    st.markdown(
        f"On **{stats['n_joined_pairs']:,} answer pairs** where human experts and GPT-4 "
        "judged the exact same pair of model responses:"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw agreement (all verdicts)", f"{stats['agreement_all']['value']:.1%}",
              help=f"95% CI {ci(stats['agreement_all'], True)}")
    c2.metric("Cohen's kappa (all)", f"{stats['kappa_all']['value']:.2f}",
              help=f"95% CI {ci(stats['kappa_all'])}")
    c3.metric("Agreement (decisive only)", f"{stats['agreement_nontie']['value']:.1%}",
              help=f"{stats['n_nontie']:,} pairs where both picked a winner; 95% CI "
                   f"{ci(stats['agreement_nontie'], True)}")
    c4.metric("Kappa (decisive only)", f"{stats['kappa_nontie']['value']:.2f}",
              help=f"95% CI {ci(stats['kappa_nontie'])}")
    st.caption(
        f"Kappa (all): {ci(stats['kappa_all'])}. Kappa (decisive): {ci(stats['kappa_nontie'])}. "
        "Landis-Koch reading: 0.21-0.40 fair, 0.41-0.60 moderate."
    )

    st.markdown("#### Is 'moderate' bad? Compare with a second human")
    st.markdown(
        f"On the **{hb['n_pairs']} pairs** that at least two experts voted on, each expert's "
        "vote is compared with every other expert's vote, and with GPT-4's. Same pairs, same "
        "unit (one judge against one human), so the numbers are directly comparable."
    )
    st.altair_chart(forest([
        row("Human vs another human", hb["human_vs_human"]["kappa"], GREEN),
        row("GPT-4 vs a human", hb["gpt4_vs_human"]["kappa"], ACCENT),
    ], "Cohen's kappa, 95% CI", [0, 0.7]), use_container_width=True)
    st.markdown(
        f"<div class='finding'><b>Finding:</b> GPT-4 agrees with a human expert at kappa "
        f"{hb['gpt4_vs_human']['kappa']['value']:.2f}; two human experts agree with each other "
        f"at {hb['human_vs_human']['kappa']['value']:.2f}. The intervals overlap almost entirely. "
        "The ceiling is the task's own ambiguity, not the judge: on this benchmark GPT-4 is "
        "roughly as reliable as adding one more human. This replicates the MT-Bench paper's "
        "conclusion from the raw votes.</div>", unsafe_allow_html=True)

    st.markdown("#### Reliability by task category")
    cats = sorted(stats["per_category"].items(), key=lambda kv: kv[1]["kappa"]["value"])
    st.altair_chart(forest(
        [row(f"{c} (n={v['n']})", v["kappa"], RED if v["kappa"]["value"] < 0.3 else BLUE)
         for c, v in cats], "Cohen's kappa vs human experts, 95% CI", [-0.1, 0.8]),
        use_container_width=True)
    gap = stats["category_gap"]
    d = gap["kappa_difference"]
    st.caption(
        f"The point estimates vary from {cats[0][1]['kappa']['value']:.2f} ({cats[0][0]}) to "
        f"{cats[-1][1]['kappa']['value']:.2f} ({cats[-1][0]}), but each category has only 10 "
        f"questions. The {gap['strong']} minus {gap['weak']} gap is {d['value']:.2f} with a "
        f"95% CI of {ci(d)}, which includes zero. Read category rankings as a hypothesis to "
        "test with more questions, not a result. STEM is the one category whose whole interval "
        "sits below 0.3."
    )

# ---- Bias analysis -------------------------------------------------------
with tab_bias:
    st.subheader("Three measurable judge behaviours")

    pos = stats["position"]
    st.markdown("#### 1. Position consistency")
    st.markdown(
        "MT-Bench asked GPT-4 to judge every pair **twice, with the answer order swapped**. "
        "If the verdict flips when only the order changes, it was position-driven. The public "
        "data records those as `tie (inconsistent)`."
    )
    b1, b2 = st.columns(2)
    b1.metric("Order-swap inconsistent judgments", f"{pos['inconsistent_count']:,} of {pos['n_judgments']:,}")
    b2.metric("Inconsistency rate", f"{pos['inconsistent_rate']['value']:.1%}",
              help=f"95% CI {ci(pos['inconsistent_rate'], True)}")

    sel = stats["selective"]
    st.markdown("#### 2. What if you abstain on those?")
    st.markdown(
        f"A common fix is to keep only verdicts that survive the order swap. That keeps "
        f"**{sel['coverage']:.0%}** of pairs and lifts raw agreement from "
        f"{stats['agreement_all']['value']:.1%} to **{sel['agreement']['value']:.1%}** "
        f"({ci(sel['agreement'], True)}), but chance-corrected kappa does not move: "
        f"{sel['kappa']['value']:.2f} vs {stats['kappa_all']['value']:.2f}. The dropped cases were "
        "ties either way, so abstention buys a cleaner-looking number, not a better judge."
    )

    v = stats["verbosity"]
    st.markdown("#### 3. Verbosity")
    st.markdown(
        f"On decisive verdicts where the two answers differ by more than "
        f"{v['length_gap_chars']} characters: how often does each judge pick the longer one? "
        "The sharpest test is the last row: on pairs where GPT-4 and the human majority picked "
        "**different** winners, which side did GPT-4 take?"
    )
    st.altair_chart(forest([
        row("Humans pick the longer answer", v["human_prefers_longer"], BLUE),
        row("GPT-4 picks the longer answer", v["gpt4_prefers_longer"], ACCENT),
        row(f"When they disagree, GPT-4 took the longer one (n={v['disagreements_gpt4_took_longer']['n']})",
            v["disagreements_gpt4_took_longer"], RED),
    ], "share of verdicts, 95% CI", [0.4, 0.9]), use_container_width=True)
    st.caption(
        f"Humans also like longer answers ({v['human_prefers_longer']['value']:.0%}), and the "
        "first two intervals overlap, so the overall preference alone does not prove a bias. "
        f"The disagreement row does: when GPT-4 overrules the humans it sides with the longer "
        f"answer {v['disagreements_gpt4_took_longer']['value']:.0%} of the time, and the whole "
        "interval sits above 50%. Observational, not a randomized length manipulation."
    )

# ---- Gallery -------------------------------------------------------------
with tab_gallery:
    st.subheader("The actual cases where GPT-4 and the humans disagreed")
    disagreements = [g for g in gallery if g["human"] != g["gpt4"]]
    agreements = [g for g in gallery if g["human"] == g["gpt4"]]
    st.markdown(
        f"Browsing **{len(disagreements)} disagreements** (and {len(agreements)} "
        "agreements for contrast) drawn from the joined dataset."
    )
    show = st.radio("Show", ["Disagreements", "Agreements"], horizontal=True)
    pool = disagreements if show == "Disagreements" else agreements
    cats_ = ["All"] + sorted({g["category"] for g in pool})
    cat = st.selectbox("Category", cats_)
    pool = [g for g in pool if cat == "All" or g["category"] == cat]

    for g in pool[:12]:
        with st.container(border=True):
            st.markdown(
                f"<div class='qbox'><b>Q{g['question_id']} ({g['category']}, turn {g['turn']}):</b> "
                f"{g['question']}</div>", unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**Model A: {g['model_a']}** ({g['len_a']:,} chars)")
                st.markdown(f"<div class='abox'>{g['answer_a']}...</div>", unsafe_allow_html=True)
            with cb:
                st.markdown(f"**Model B: {g['model_b']}** ({g['len_b']:,} chars)")
                st.markdown(f"<div class='abox'>{g['answer_b']}...</div>", unsafe_allow_html=True)
            v1, v2 = st.columns(2)
            with v1:
                st.markdown(f"Human experts ({g['n_human_votes']} votes): " + pill(g["human"]),
                            unsafe_allow_html=True)
            with v2:
                note = " (flipped when the order was swapped)" if g.get("gpt4_raw") == "tie (inconsistent)" else ""
                st.markdown("GPT-4 judge: " + pill(g["gpt4"]) + note, unsafe_allow_html=True)

# ---- Methodology ---------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
### What is this?

**JudgeLab** measures the reliability of an LLM acting as a judge, the question every
RLAIF pipeline and automated eval quietly depends on. Teams say "we used GPT-4 as a judge";
JudgeLab asks **how good a judge is it, compared with what, and how sure are we?**

### The data (all real, all public)

- Source: `lmsys/mt_bench_human_judgments` on Hugging Face, released with the MT-Bench
  paper (Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena").
- **{stats['n_human_votes']:,} pairwise votes by human experts** on pairs of real model
  answers (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B).
- **{stats['n_gpt4_judgments']:,} GPT-4 pairwise judgments**, each made in both answer orders.
- Joined on identical (question, model pair, turn): **{stats['n_joined_pairs']:,} pairs**.

### Statistics

- **Cohen's kappa** (three classes: A wins, B wins, tie) between GPT-4 and the human
  majority, on all verdicts and on decisive (non-tie) verdicts.
- **95% confidence intervals** by percentile bootstrap, **resampling whole questions**
  (2,000 draws; 1,000 per category). MT-Bench has 80 questions and each contributes many
  pairs; a pair-level bootstrap would treat correlated pairs as independent and report
  intervals that are too narrow.
- **Human baseline:** on pairs with two or more expert votes, every expert-expert vote pair
  versus every expert-GPT-4 pair. Same items, same unit.
- **Position consistency** from MT-Bench's both-orders protocol; **selective judging**
  (abstain on inconsistent verdicts); **verbosity** overall and in disagreements.

### Reproduce

```
python prepare_data.py   # download the public data (no key), write the compact file
python analyze.py        # every number, offline, deterministic seed
python analyze.py --check  # CI: fails if data/stats.json drifts from the data
```

### Honest limits

- Human majorities on few votes per pair are themselves noisy; ties are analysed both
  included and excluded, and the human baseline uses only multi-vote pairs.
- Per-category intervals are wide (10 questions each). Category rankings are hypotheses.
- Verbosity is an observational association, not a randomized experiment.
- One judge (GPT-4, 2023) on one benchmark. The method is the transferable part.
"""
    )
