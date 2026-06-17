<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator

[![Leaderboard](https://img.shields.io/badge/Leaderboard-github.io-green?link=https://layer6ai-labs.github.io/RankJudge/)](https://layer6ai-labs.github.io/RankJudge/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?link=https://huggingface.co/datasets/Layer6/RankJudge)](https://huggingface.co/datasets/Layer6/RankJudge)
[![Paper](https://img.shields.io/badge/arXiv-2605.21748-b31b1b?logo=arxiv&logoWidth=10&link=https://arxiv.org/abs/2605.21748)](https://arxiv.org/abs/2605.21748)

As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q&A tasks that do not match the complexity of multi-turn conversations. We introduce **RankJudge**, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate **21 frontier LLM judges**, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm.

Three domains:
- `ml`: CS papers from [RPC-Bench](https://github.com/RPC-Bench/PRC-Bench).
- `med`: medical papers from [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA).
- `fin`: S&P 500 10-K filings from [sp500-edgar-10k](https://huggingface.co/datasets/jlohding/sp500-edgar-10k).

## Two ways to use this repo

**Mode A: Rank from the released dataset (default).** Skip generation entirely: download the published 652-pair / 13,692-match evaluation slice from `Layer6/RankJudge` on Hugging Face and compute Bradley-Terry rankings directly. No API calls.

**Mode B: Regenerate from scratch.** Run the full pipeline (preprocess → pairs → verify → matches → metrics) to produce your own pairs and judge them with any model roster you like. Requires an OpenRouter API key.

## Setup

```bash
pip install requests pandas pyarrow datasets numpy
pip install streamlit   # optional, only needed for the data explorer
```

Mode B additionally needs an [OpenRouter](https://openrouter.ai) API key in `api_key.json` at the repo root:

```json
{"api_key": "sk-or-v1-..."}
```

## Mode A: Rank from the released dataset

One command:

```bash
bash scripts/run.sh a
```

(equivalently `cd code && python rank_from_hf.py`; the runner just `exec`s that, forwarding any extra flags.)

This downloads `Layer6/RankJudge` (652 pairs + 13,692 matches) into `../data`, materializes it as organized JSON in `../outputs/mode_a/` (`pairs.json`, `matches.json`), then computes `../outputs/mode_a/metrics.json` directly from that local `matches.json`: Bradley-Terry rankings for all 21 judges, broken down by domain, assistant weakness, and user behavior. The released matches are already the published evaluation slice, so no further filtering is applied (`--no-top-removed` is the default in this mode). Mode A and Mode B write to separate directories (`outputs/mode_a/` vs `outputs/mode_b/`), so they never collide.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--repo` | `Layer6/RankJudge` | HF dataset id |
| `--split` | `train` | Split to load |
| `--cache-dir` | `../data` | HF datasets cache dir |
| `--out-dir` | `../outputs/mode_a` | Where `pairs.json` / `matches.json` are materialized |
| `--metrics-out` | `../outputs/mode_a/metrics.json` | Metrics output path |
| `--init-elo` | 1500 | Starting Elo (BT anchor) |

## Mode B: Regenerate from scratch

### Pipeline

```
raw data -> [preprocessing/] -> data/input/<domain>.json
                                      |
                                      v
                                 [pairs.py] -> pairs.json
                                      |
                                      v
                                 [verify.py] -> verification.json, pairs_filtered.json
                                      |
                                      v
                                 [matches.py] -> matches.json
                                      |
                                      v
                                 [metrics.py] -> metrics.json
```

1. **Preprocess** (`preprocessing/{ml,med,fin}.py`): per-domain scripts that normalize raw data into a shared `{id, context}` format. Run once per domain.
2. **Generate pairs** (`pairs.py`): for each item, sample a user behavior, an assistant weakness, and a round count. Produce a good conversation and a bad one (with the weakness injected into a single round). A/B order is randomized.
3. **Verify** (`verify.py`): three-layer check on each pair: *coherence* (is the plan internally consistent?), *adherence* (did each conversation follow its plan, with the flaw landing in the right round?), and *grounding* (are assistant claims supported by the source? The flawed round is excluded from the bad convo's grounding rate, since its claim may be intentionally ungrounded). Writes `verification.json` and emits `pairs_filtered.json`, the pairs that passed all three checks.
4. **Run matches** (`matches.py`): run 21 judge models on every pair in `pairs_filtered.json`. Each judge predicts verdict (A/B), worst round, and weakness type. Correctness requires all three to match ground truth. Calls `metrics.py` once finished.
5. **Metrics** (`metrics.py`): rate judges and pairs with Bradley-Terry, broken down by assistant weakness, user behavior, and domain.

### Usage

One command runs all five stages end to end:

```bash
bash scripts/run.sh b      # or just `bash scripts/run.sh` (Mode B is the default)
```

**First run vs. subsequent runs.** The stage flags in `run.sh` ship with `RUN_PREPROCESS=1`, so the first run downloads the raw sources and builds `data/input/<domain>.json`. Preprocessing is cached, so on later runs you can set `RUN_PREPROCESS=0` to skip straight to `pairs.py` (saves the download + normalization step). The other stage flags (`RUN_PAIRS`, `RUN_VERIFY`, `RUN_MATCHES`, `RUN_METRICS`) work the same way: turn a stage off once its output exists.

To drive a single stage yourself, run the scripts directly. They all run from `code/` and reference paths relative to it (`../outputs/`, `../data/`, `../api_key.json`):

```bash
cd code

# 1. Preprocess each data source (run once)
python preprocessing/ml.py
python preprocessing/med.py
python preprocessing/fin.py

# 2. Generate pairs (all 3 domains by default)
python pairs.py --n-samples 100 --workers 50
python pairs.py --n-samples 100 --workers 50 --dataset ml fin   # subset

# 3. Verify, filter, and write pairs_filtered.json
python verify.py --workers 20 --resume

# 4. Run the matches (and compute metrics afterwards)
python matches.py --workers 50 --resume

# 5. Recompute metrics without rerunning judges (optional)
python metrics.py
```

### Flags

| Script | Flag | Default | Description |
|---|---|---|---|
| `pairs.py` | `--dataset` | `ml med fin` | Which domains to process |
| `pairs.py` | `--n-samples` | 10 | Items per domain |
| `pairs.py` | `--model` | `openai/gpt-5.5` | Generator model |
| `pairs.py` | `--uniform` | on | Sample taxonomy keys uniformly. Use `--no-uniform` to follow `sampler.py` `DISTRIBUTIONS` weights. |
| `verify.py` | `--model` | `openai/gpt-5.5` | Verifier model |
| `verify.py` | `--workers` | 20 | Parallel verifier calls |
| `matches.py` | `--workers` | 50 | Parallel judge calls |
| `matches.py` | `--resume` | off | Resume from existing `matches.json` |
| `matches.py` | `--max-tokens` | 32768 | Max tokens per judge response |
| `metrics.py` | `--init-elo` | 1500 | Starting Elo rating (BT anchor) |
| `metrics.py` | `--top-pct` | 0.05 | Fraction of top-Elo pairs to drop for the `top_removed` slice |

## Explore the pairs

A Streamlit UI shows the pairs round by round with per-judge predictions. Pick the mode to match where the data came from:

```bash
cd code
streamlit run explorer.py              # Mode A: the released HF pairs (default)
streamlit run explorer.py -- --mode b  # Mode B: your local pipeline output
```

The `--` separator passes flags to the script rather than to Streamlit. Mode A hides the verification tab, since the HF release ships no verification records; Mode B adds the verification tab and the filtered-vs-all toggle.

## Citation

```bibtex
@article{tang2026rankjudge,
  title={RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator},
  author={Tang, Zhenwei and Liu, Zhaoyan and Hosseinzadeh, Rasa and Wu, Tongzi and Golestan, Keyvan and Cresswell, Jesse C},
  journal={arXiv:2605.21748},
  year={2026}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
