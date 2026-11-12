# RepairJudge

**Can LLM judges tell appropriate revision from capitulation when users push back?**

Final project, Georgia Institute of Technology.
Team: Siwan Yang, Ruby Yu, Siang-Yu Cheng, Wei-Po Lin

> Built on [RankJudge](https://github.com/layer6ai-labs/RankJudge) (Tang et al., 2026), MIT License.
> The original README is kept at [`docs/RANKJUDGE_README.md`](docs/RANKJUDGE_README.md).

---

## Overview

When a user challenges an assistant's answer ("Shouldn't it be X?"), the assistant can fail in two ways:

- **Capitulation**: the assistant was right, the user was wrong, and the assistant gives in.
- **Failure to revise**: the assistant was wrong, the user was right, and the assistant refuses to change.

We test whether **LLM judges** can detect these failures. Following RankJudge, we build pairs of
multi-turn conversations that are identical except for the assistant's final turn. One conversation
handles the correction well; the other contains exactly one of the two failures. Judges must pick
the better conversation, the flawed turn, and the error type.

**Research questions**

| | Question |
|---|---|
| RQ1 | Can judges identify capitulation to incorrect user corrections? |
| RQ2 | Can judges identify failure to revise after correct user corrections? |
| RQ3 | Does providing the gold answer improve judge accuracy? |

**Data:** 300 problems from [GSM8K](https://huggingface.co/datasets/openai/gsm8k), one
wrong-correction pair and one right-correction pair per problem.

## Status

- [x] Fork RankJudge and reproduce Mode A
- [ ] GSM8K preprocessing
- [ ] Templated pair generation (turns 1–3 templated, turn 4 generated)
- [ ] Numeric verification against GSM8K gold answers
- [ ] Judge prompts with new error types and gold-answer condition
- [ ] Metrics: joint accuracy, winner accuracy, 95% CIs, baselines
- [ ] Poster

## Repository structure

```
RepairJudge/
├── code/
│   ├── preprocessing/
│   │   ├── ml.py, med.py, fin.py   # original RankJudge domains
│   │   └── gsm8k.py                # NEW (planned)
│   ├── sampler.py                  # MODIFIED: add capitulation / failure_to_revise
│   ├── pairs.py                    # MODIFIED: templated GSM8K conversations
│   ├── verify.py                   # MODIFIED: numeric answer check
│   ├── matches.py                  # MODIFIED: new error types, --gold condition
│   ├── metrics.py                  # MODIFIED: accuracies, CIs, baselines
│   ├── rank_from_hf.py             # original (Mode A)
│   └── explorer.py                 # original Streamlit viewer
├── scripts/run.sh
├── docs/RANKJUDGE_README.md        # original README
├── requirements.txt
└── LICENSE
```

## Setup

**1. Clone**

```bash
git clone https://github.com/YOUR_ORG/RepairJudge.git
cd RepairJudge
git remote add upstream https://github.com/layer6ai-labs/RankJudge.git
```

**2. Environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. API key** (needed for generation and judging, not for Mode A)

Create `api_key.json` in the repo root with your own [OpenRouter](https://openrouter.ai) key:

```json
{"api_key": "sk-or-v1-..."}
```

Then confirm it is ignored by git:

```bash
git status    # api_key.json must NOT appear
```

⚠️ Never commit API keys. If one is pushed by accident, revoke it on OpenRouter immediately.

## Quick start: reproduce the RankJudge baseline

No API calls, no cost:

```bash
bash scripts/run.sh a
```

Outputs go to `outputs/mode_a/` (`pairs.json`, `matches.json`, `metrics.json`).
To browse the pairs:

```bash
cd code && streamlit run explorer.py
```

## Our pipeline (planned, commands will change)

```bash
cd code
python preprocessing/gsm8k.py --n 300           # build data/input/gsm8k.json
python pairs.py --dataset gsm8k                 # generate conversation pairs
python verify.py --resume                       # numeric + style checks
python matches.py --resume                      # judges, dialogue-only condition
python matches.py --resume --gold               # judges, with gold answer
python metrics.py                               # accuracies, CIs, baselines
```

## Team workflow

- Work on a branch, never directly on `main`.
- Open a pull request when a piece works; one teammate reviews before merging.
- Keep generated outputs (`outputs/`, `data/`) out of git unless we agree to share a specific file.

| Member | Area | Branch |
|---|---|---|
| Ruby | Research questions, repair strategies, judge prompts | `judge-prompts` |
| Siwan | GSM8K dataset, baselines, metrics | `gsm8k-preprocessing` |
| Will | Conversation generation, data validation | `pair-templates` |
| Paul | Verification, risk and fallback, experimental setup | `verify-numeric` |

## Acknowledgments and citation

This project adapts the RankJudge pipeline. If you use this code, please cite the original work:

```bibtex
@article{tang2026rankjudge,
  title={RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator},
  author={Tang, Zhenwei and Liu, Zhaoyan and Hosseinzadeh, Rasa and Wu, Tongzi and Golestan, Keyvan and Cresswell, Jesse C},
  journal={arXiv preprint arXiv:2605.21748},
  year={2026}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

The user repair strategies follow Lachenmaier, Bultmann, and Zarrieß (ACL 2026),
*Talking to a Know-It-All GPT or a Second-Guesser Claude? How Repair reveals distinct Multi-Turn Behavior in LLMs*.

## License

MIT. The original RankJudge code is copyright Layer 6 AI; see [`LICENSE`](LICENSE).
Modifications are copyright the RepairJudge authors.
