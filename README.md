# JudgeLab - How Much Can You Trust an LLM as a Judge?

**GPT-4-as-a-judge, measured against 3,355 real human expert votes: 87% agreement sounds great until you see Cohen's kappa of 0.46, verdicts that flip 15.8% of the time when the answers swap places, and reliability that collapses on writing tasks.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit&logoColor=white)
![Data](https://img.shields.io/badge/data-real_MT--Bench_human_votes-brightgreen)
![No API key](https://img.shields.io/badge/API_key-not_required-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<!--
Add a screenshot/GIF after deploying: docs/demo.png
-->

> **What this is:** every RLAIF pipeline and automated eval quietly assumes an LLM judge
> can be trusted. JudgeLab measures that assumption on real, public data - the MT-Bench
> human study - and shows exactly where the judge holds up and where it breaks.
> Companion project to [TrajLens](https://github.com/PSPMANI/trajlens): TrajLens grades
> agents with deterministic rubrics; JudgeLab measures the automated judge itself.

---

## The headline numbers (all computed from real data)

On 1,232 answer pairs where human experts and GPT-4 judged the exact same pair:

| Metric | Value | Reading |
|---|---|---|
| Raw agreement (all verdicts) | 68.4% | looks fine... |
| **Cohen's kappa (all)** | **0.41** | ...but only moderate once chance is removed |
| Agreement (decisive verdicts) | 87.1% | the number people quote |
| **Kappa (decisive)** | **0.46** | the number they should quote |
| **Order-swap inconsistency** | **15.8%** (380 verdicts) | the verdict flipped when answers swapped seats |
| Verbosity preference (GPT-4) | 73.1% picks the longer answer | humans: 67.6% - a measurable thumb on the scale |
| Weakest category | writing, kappa 0.18 | vs math at 0.49 - reliability is not uniform |

## What it does

- **Reliability report:** raw agreement and Cohen's kappa between GPT-4 and the human
  majority, on all verdicts and decisive-only, plus a per-category kappa chart that
  shows the judge is weakest exactly where quality is most subjective (writing,
  roleplay, stem) and strongest where answers are checkable (math, reasoning).
- **Disagreement gallery:** the actual cases - question, both model answers, the human
  vote, and the GPT-4 verdict side by side - where the judge got it wrong.
- **Bias analysis:** position bias measured from MT-Bench's own both-orders judging
  protocol (15.8% of verdicts flipped with seat order), and verbosity preference
  compared judge-vs-human on the same length-mismatched pairs.
- **Methodology tab:** every formula, every source, and the honest limits.

## The data (real, public, keyless)

- `lmsys/mt_bench_human_judgments` on Hugging Face, released with the MT-Bench paper
  (Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena").
- 3,355 pairwise votes by human experts on answers from 6 real models
  (GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, LLaMA-13B).
- 2,400 GPT-4 pairwise judgments on the same pairs, each judged in both answer orders.
- Joined on identical (question, model pair, turn): 1,232 directly comparable pairs.

No LLM API is called anywhere: the judge being measured already did its judging in the
published dataset. The app runs entirely on two small committed JSON files.

## Reproduce every number

```
python prepare_data.py
```

downloads the public dataset from the Hugging Face datasets server (no key, polite
rate-limited, resumable) and regenerates `data/stats.json` and `data/gallery.json`.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)

1. Push to a public GitHub repo.
2. share.streamlit.io -> New app -> pick the repo, main file `app.py`, Python 3.11.
3. Done - the app is static-data driven, so it never bills and never breaks.

## Why this matters

If you train against AI feedback (RLAIF) or rank models with an LLM judge, the judge's
bias becomes your model's bias. "We used GPT-4 as a judge" is a methods sentence;
"our judge agrees with human experts at kappa 0.46, is position-consistent 84% of the
time, and is weakest on writing tasks" is an evaluation. This project is the second
sentence.

## Honest limits

- Human majorities on few votes per pair are themselves noisy; ties are analyzed both
  included and excluded.
- Verbosity preference is an observational association, not a randomized experiment.
- This measures one judge (GPT-4, 2023) on one benchmark. The numbers are dated by
  design - the method is the transferable part.
