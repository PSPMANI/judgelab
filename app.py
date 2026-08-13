"""JudgeLab - how much can you trust an LLM as a judge?

Measures GPT-4-as-a-judge against 3.3k real human expert votes from the
MT-Bench study (lmsys/mt_bench_human_judgments): raw agreement, Cohen's kappa,
position consistency, verbosity bias, and per-category reliability - plus a
gallery of the actual cases where the judge and the humans disagreed.

All data is real, public, and precomputed: no API key, zero cost, reproducible
with `python prepare_data.py`.
"""
import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="JudgeLab - LLM-as-a-Judge Reliability",
                   layout="wide")

DATA = pathlib.Path(__file__).parent / "data"

GREEN = "#22C55E"
RED = "#EF4444"
ACCENT = "#8B5CF6"
BLUE = "#3B82F6"

CSS = """
<style>
.muted {color:#94A3B8;font-size:0.85rem;}
.verdict {display:inline-block;padding:3px 12px;border-radius:12px;color:white;font-weight:700;font-size:0.85rem;}
.qbox {padding:10px 14px;background:rgba(139,92,246,0.08);border-left:4px solid #8B5CF6;border-radius:6px;margin-bottom:10px;}
.abox {padding:10px 14px;background:rgba(148,163,184,0.07);border-radius:6px;font-size:0.9rem;}
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


stats, gallery = load()

st.sidebar.markdown("## JudgeLab")
st.sidebar.caption("LLM-as-a-Judge Reliability Lab")
st.sidebar.markdown(
    f"Built on **{stats['n_human_votes']:,} real human expert votes** and "
    f"**{stats['n_gpt4_judgments']:,} real GPT-4 judgments** from the MT-Bench "
    "study (lmsys/mt_bench_human_judgments)."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Everything is precomputed from the public dataset: no API key, zero cost. "
    "Run `python prepare_data.py` to regenerate every number from source."
)
st.sidebar.markdown(
    "<span class='muted'>Companion project to TrajLens: TrajLens grades agents "
    "with deterministic rubrics; JudgeLab measures the automated judge itself.</span>",
    unsafe_allow_html=True,
)

tab_overview, tab_gallery, tab_bias, tab_about = st.tabs(
    ["Reliability report", "Disagreement gallery", "Bias analysis", "Methodology"]
)

# ---- Overview ------------------------------------------------------------
with tab_overview:
    st.subheader("Can you trust GPT-4 as a judge? Measured, not assumed.")
    st.markdown(
        f"On **{stats['n_joined_pairs']:,} answer pairs** where human experts and "
        "GPT-4 judged the exact same pair of model responses:"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw agreement (all verdicts)", f"{stats['agreement_all']*100:.1f}%")
    c2.metric("Cohen's kappa (all)", f"{stats['kappa_all']:.2f}")
    c3.metric("Agreement (decisive only)", f"{stats['agreement_nontie']*100:.1f}%",
              help=f"{stats['n_nontie']:,} pairs where both picked a winner")
    c4.metric("Kappa (decisive only)", f"{stats['kappa_nontie']:.2f}")

    st.caption(
        "Kappa corrects agreement for chance. Landis-Koch reading: below 0.20 slight, "
        "0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial. An unmeasured judge "
        "is an unbudgeted risk in any RLAIF or eval pipeline."
    )

    st.markdown("#### Where the judge is weakest: reliability by task category")
    rows = [{"category": c, "kappa": v["kappa"], "agreement": v["agreement"], "pairs": v["n"]}
            for c, v in stats["per_category"].items()]
    df = pd.DataFrame(rows).sort_values("kappa")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("kappa:Q", title="Cohen's kappa vs human experts"),
            y=alt.Y("category:N", sort="x", title=None),
            color=alt.condition(alt.datum.kappa < 0.4,
                                alt.value(RED), alt.value(GREEN)),
            tooltip=["category", "kappa", "agreement", "pairs"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Judge reliability is not uniform: it varies sharply by task type. A team that "
        "trusts one global agreement number is flying blind on its weakest categories."
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
    cats = ["All"] + sorted({g["category"] for g in pool})
    cat = st.selectbox("Category", cats)
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
                st.markdown(
                    f"Human experts ({g['n_human_votes']} votes): " + pill(g["human"]),
                    unsafe_allow_html=True)
            with v2:
                st.markdown("GPT-4 judge: " + pill(g["gpt4"]), unsafe_allow_html=True)

# ---- Bias analysis -------------------------------------------------------
with tab_bias:
    st.subheader("Two measurable judge biases")

    st.markdown("#### 1. Position consistency")
    st.markdown(
        "MT-Bench asked GPT-4 to judge every pair **twice, with the answer order "
        "swapped**. If the verdict flips when only the order changes, the judgment "
        "was position-driven, not quality-driven. Those cases are recorded as "
        "inconsistent in the public data."
    )
    b1, b2 = st.columns(2)
    b1.metric("Order-swap inconsistent judgments",
              f"{stats['gpt4_inconsistent_count']:,}")
    b2.metric("Inconsistency rate",
              f"{stats['gpt4_inconsistent_rate']*100:.1f}%")
    st.caption(
        "Every one of these is a verdict that changed because the answers swapped "
        "seats. A human grader whose opinion flipped on seating order would not "
        "keep the job."
    )

    st.markdown("#### 2. Verbosity preference")
    st.markdown(
        "Among decisive verdicts on pairs where one answer is meaningfully longer "
        "(over 50 characters difference): how often does each judge pick the longer answer?"
    )
    vdf = pd.DataFrame([
        {"judge": "Human experts", "prefers longer": stats["verbosity_pref_human"],
         "n": stats["verbosity_n_human"]},
        {"judge": "GPT-4 judge", "prefers longer": stats["verbosity_pref_gpt4"],
         "n": stats["verbosity_n_gpt4"]},
    ])
    chart = (
        alt.Chart(vdf)
        .mark_bar()
        .encode(
            x=alt.X("prefers longer:Q", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]), title="picks the longer answer"),
            y=alt.Y("judge:N", title=None),
            color=alt.Color("judge:N", legend=None,
                            scale=alt.Scale(range=[BLUE, ACCENT])),
            tooltip=["judge", "prefers longer", "n"],
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)
    delta = (stats["verbosity_pref_gpt4"] - stats["verbosity_pref_human"]) * 100
    st.caption(
        f"GPT-4 picks the longer answer {delta:+.1f} percentage points more often than "
        "human experts do on the same pairs - a measurable thumb on the scale for length."
    )

# ---- Methodology ---------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
### What is this?

**JudgeLab** measures the reliability of an LLM acting as a judge - the exact
question every RLAIF pipeline and automated eval quietly depends on. Teams say
"we used GPT-4 as a judge"; JudgeLab asks: **how good a judge is it, exactly?**

### The data (all real, all public)

- Source: `lmsys/mt_bench_human_judgments` on Hugging Face, released with the
  MT-Bench paper (Zheng et al., 2023, "Judging LLM-as-a-Judge").
- **{stats['n_human_votes']:,} pairwise votes by human experts** on pairs of real
  model answers (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B).
- **{stats['n_gpt4_judgments']:,} GPT-4 pairwise judgments** on the same answer
  pairs, each judged in both answer orders.
- Joined on identical (question, model pair, turn): **{stats['n_joined_pairs']:,} pairs**
  where humans and GPT-4 judged exactly the same thing.

### What is computed

- **Raw agreement and Cohen's kappa** between GPT-4 and the human majority,
  on all verdicts and on decisive (non-tie) verdicts. Kappa corrects for chance
  agreement - the honest statistic, and the same one used for human
  inter-annotator reliability.
- **Position consistency**: MT-Bench judged every pair in both answer orders;
  verdicts that flipped with the order are recorded as inconsistent. That is
  position bias, measured - not anecdotes.
- **Verbosity preference**: on length-mismatched pairs, how often each judge
  picks the longer answer. The gap between GPT-4 and humans is the bias.
- **Per-category reliability**: kappa recomputed per MT-Bench category
  (writing, roleplay, reasoning, math, coding, extraction, stem, humanities),
  because a single global number hides the judge's weakest domains.

### Why it matters

If you train against AI feedback (RLAIF) or rank models with an LLM judge, the
judge's bias becomes your model's bias. Before trusting a judge, measure it -
and read its failure cases, which is what the Disagreement gallery is for.

### Reproduce every number

```
python prepare_data.py
```

downloads the public dataset (no key) and regenerates `data/stats.json` and
`data/gallery.json` from source.

### Honest limits

- Human majority votes on few votes per pair can themselves be noisy; ties are
  analyzed both included and excluded.
- Verbosity preference is an observational association, not a randomized test.
- This measures one judge (GPT-4, 2023) on one benchmark; the method, not the
  specific numbers, is the transferable part.
"""
    )
