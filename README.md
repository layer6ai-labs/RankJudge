# RepairJudge

**Can LLM judges tell appropriate revision from capitulation when users push back?**

- CS 6604 Final project
- Georgia Institute of Technology.
- **Team Member:** Siwan Yang, Ruby Yu, Siang-Yu Cheng, Wei-Po Lin

> Built on [RankJudge](https://github.com/layer6ai-labs/RankJudge) (Tang et al., 2026), MIT License.

---

## Overview

When a user challenges an assistant's answer ("Shouldn't it be X?"), the assistant can fail in two ways:

- **Capitulation**: the assistant was right, the user was wrong, and the assistant gives in.
- **Failure to revise**: the assistant was wrong, the user was right, and the assistant refuses to change.

We build pairs of conversations that are identical except for the assistant's last reply (one
good, one with one of these failures) and test whether **LLM judges** pick the right one.

| | Research question |
|---|---|
| RQ1 | Can judges identify capitulation to incorrect user corrections? |
| RQ2 | Can judges identify failure to revise after correct user corrections? |
| RQ3 | Does providing the gold answer improve judge accuracy? |

**Data:** 300 math word problems from [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

---

## Quick start

Requires Python 3.10+. Run every command from the repo root (`RepairJudge/`).

### 1. Clone

```bash
git clone https://github.com/siwany/RepairJudge.git
cd RepairJudge
```

### 2. Create a virtual environment and install packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the data (free, no API key)

```bash
cd code && python preprocessing/gsm8k.py --n 300 && cd ..
```

This downloads GSM8K from Hugging Face, samples 300 problems (fixed seed, so everyone gets the
same ones), and writes **`data/input/gsm8k.json`**. The `data/` folder is git-ignored, so each
teammate runs this once.

### 4. Check the code works (free, no API key)

```bash
python scripts/smoke_test_repair.py      # should end with "SMOKE TEST PASSED"
```

This runs the whole pipeline with fake model answers.

### 5. Run the pipeline on PACE (free GPUs)

We generate the dataset and run all five judges on Georgia Tech's [PACE](https://pace.gatech.edu)
cluster. A laptop is too slow for a strong generator, and PACE costs nothing. **One person runs
this once and shares the results** (step 7).

| Role | Model | Why |
|---|---|---|
| Generator + verifier | `gpt-oss:120b` | strong, fits on one 80 GB GPU, and not from the same family as any judge |
| Judges | `llama3.1:8b`, `llama3.2:3b`, `gemma3:4b`, `mistral:7b`, `qwen2.5:7b` (`JUDGE_SETS["local"]` in `code/models.py`) | five small open models from different families |

The models are served on the GPU node by [Ollama](https://ollama.com), which runs as a normal
program in your own folder. It can keep several models available behind one address, so the
pipeline switches between the generator and the five judges without restarting anything.

> PACE details (cluster name, GPU flags, modules, storage paths) change over time and differ
> between clusters. Check the commands below against the [PACE docs](https://docs.pace.gatech.edu)
> before you run them.

**5.1 Log in and get a GPU node.** For a course account this is usually the ICE cluster:

```bash
ssh <gt-username>@login-ice.pace.gatech.edu
salloc --gres=gpu:H100:1 --mem=64G --cpus-per-task=8 -t 4:00:00    # interactive, 4 hours
```

**5.2 Install the model server in your scratch space** (once). There is no `sudo` on PACE, so
unpack the Linux build into your own folder. Your home folder is small, so keep models in scratch:

```bash
mkdir -p ~/scratch/ollama && cd ~/scratch/ollama
curl -fL https://ollama.com/download/ollama-linux-amd64.tgz | tar xz
# If that file name has changed, follow "Manual install" at https://docs.ollama.com/linux
```

**5.3 Start the server** (every time you get a new node):

```bash
export PATH=~/scratch/ollama/bin:$PATH
export OLLAMA_MODELS=~/scratch/ollama-models
export OLLAMA_CONTEXT_LENGTH=16384      # the judge prompt is long
export OLLAMA_NUM_PARALLEL=4
ollama serve > ~/scratch/ollama.log 2>&1 &
```

**5.4 Download the models** (once; about 65 GB for the generator plus about 25 GB for the judges):

```bash
ollama pull gpt-oss:120b
ollama pull llama3.1:8b && ollama pull llama3.2:3b && ollama pull gemma3:4b \
  && ollama pull mistral:7b && ollama pull qwen2.5:7b
ollama run llama3.2:3b "What is 2+2?"   # quick check that the server answers
```

**5.5 Set up the repo on PACE** (once). Use any Python 3.10+ (`module avail python` lists them):

```bash
cd ~/scratch && git clone https://github.com/siwany/RepairJudge.git && cd RepairJudge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**5.6 Test with 1 problem, then a 10-problem pilot:**

```bash
N_PROBLEMS=1 OUT=../outputs/test_pace bash scripts/run_repair.sh
N_PROBLEMS=10 OUT=../outputs/pilot bash scripts/run_repair.sh
```

The script generates the pairs, verifies them, runs the 5 judges in both conditions, and computes
metrics. Copy the pilot to your laptop (step 7) and read it in the explorer (step 6). Before the
full run, check that:
- most pairs passed verification;
- the good and bad replies read naturally and differ only in the intended way;
- the terminal did not show many `ERROR` or `parse_fail` lines. Many errors usually mean the
  server is not running (step 5.3) or a model was not pulled (step 5.4).

If `gpt-oss:120b` keeps failing with JSON or `parse_fail` errors, use `GEN_MODEL=qwen3:32b`
(pull it first). It shares a family with the `qwen2.5-7b` judge, so mention that in the paper or
drop that judge.

**5.7 Full run (300 problems) as a batch job.** A batch job keeps running after you log out.
Save this as `~/scratch/repair.sbatch` and submit it with `sbatch ~/scratch/repair.sbatch`:

```bash
#!/bin/bash
#SBATCH -J repairjudge
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.log

export PATH=~/scratch/ollama/bin:$PATH
export OLLAMA_MODELS=~/scratch/ollama-models
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=4
ollama serve > ~/scratch/ollama.log 2>&1 &
sleep 20

cd ~/scratch/RepairJudge && source .venv/bin/activate
bash scripts/run_repair.sh               # results in outputs/repair/
```

Check progress with `squeue -u $USER` and `tail -f repairjudge-<jobid>.log`. If the job hits its
time limit, submit it again: every stage resumes where it stopped.

To generate the dataset only (no judges yet), set `RUN_JUDGE_DIALOGUE=0`, `RUN_JUDGE_GOLD=0` and
`RUN_METRICS=0` at the top of `scripts/run_repair.sh`.

### 6. Browse the data (explorer)

On your laptop:

```bash
streamlit run code/explorer.py
```

Opens http://localhost:8501. In the sidebar, pick the **data source**: one of our runs under
`outputs/` (for example `outputs/repair`) or the original **RankJudge (Hugging Face)** dataset.
For our runs you can switch between all pairs and the pairs that passed verification, filter by
error type (`capitulation` / `failure_to_revise`), step through pairs with prev/next, and read
each pair's conversations, verification checks and, once the judges have run, every judge's
answer (dialogue-only or with-gold condition). Stop it with `Ctrl+C`.

To read the 300 source GSM8K problems themselves, use `streamlit run code/data_viewer.py`.

### 7. Share the results with the team

Run this on your laptop to copy the results from PACE:

```bash
scp -r <gt-username>@login-ice.pace.gatech.edu:~/scratch/RepairJudge/outputs/repair outputs/
cp data/input/gsm8k.json outputs/repair/
cd outputs && zip -r repairjudge_gsm8k_v1.zip repair && cd ..
```

Upload the zip to the team Google Drive folder (share as Viewer) with a note of the generator
model and date. Teammates unzip it into `outputs/` and open it in the explorer. It contains the
dataset (`pairs_filtered.json`), the verification results and the judge results, so nobody has
to rerun anything. If you regenerate, save it as `_v2` instead of replacing `_v1`.

### Optional: generate with the OpenRouter API

If PACE is not available, the generator can run on [OpenRouter](https://openrouter.ai) instead
(the judges still run on PACE).

1. Make a key at [openrouter.ai](https://openrouter.ai) and add some credit.
2. Create `api_key.json` in the repo root, on the machine that runs the pipeline:

   ```json
   {"api_key": "sk-or-v1-..."}
   ```

3. Run `git status` and make sure `api_key.json` is **not** listed (it is in `.gitignore`).
   ⚠️ Never commit a key. If one is pushed by accident, revoke it on OpenRouter right away.
4. Run with `GEN_BACKEND=openrouter`. The default API generator is `google/gemma-4-31b-it`
   (about $1 for 300 problems; about 85% of its pairs passed verification in our pilot):

   ```bash
   GEN_BACKEND=openrouter N_PROBLEMS=10 OUT=../outputs/pilot_api bash scripts/run_repair.sh
   ```

   `GEN_BACKEND` controls pair generation and verification; `JUDGE_BACKEND` controls the judges.
   Both default to `BACKEND` (`local`). `BACKEND=openrouter` would also put the judges on the API
   (`JUDGE_SETS["openrouter"]`), which costs far more (about $40 for the full run).

---

## Workflow

```
 ① preprocessing/gsm8k.py   download + sample 300 GSM8K problems      → data/input/gsm8k.json
          │
 ② pairs.py                 build 2 conversation pairs per problem    → outputs/.../pairs.json
          │
 ③ verify.py                check numbers vs. the gold answer;       → pairs_filtered.json
          │                 keep only clean pairs
 ④ matches.py  (run twice)  5 judges: dialogue only / with gold answer → matches_dialogue.json
          │                                                              matches_gold.json
 ⑤ metrics.py --repair      accuracies, 95% CIs, baselines            → metrics.json, metrics_table.csv
```

`scripts/run_repair.sh` runs ① to ⑤ in order.

| Stage | Uses an LLM? | What the LLM does |
|---|---|---|
| ① preprocessing | no | – |
| ② pairs | yes | writes a wrong solution that reaches the wrong value W (1 call per problem) and the good and bad last reply (1 call per pair) |
| ③ verify | partly | numeric checks are plain code; one audit call per pair (can be turned off) |
| ④ matches | yes | the judges being tested |
| ⑤ metrics | no | – |

### What one pair looks like

Each conversation has 4 turns. Turns 1–3 are the same in both conversations; only turn 4 differs.
G = correct answer, W = wrong answer. W is picked by code, not by the model: usually an intermediate
result of the GSM8K solution (as if the assistant stopped a step early), otherwise a simple slip
such as 2×G. The model only writes a solution that reaches W.

| Turn | Role | `capitulation` pair (RQ1): assistant right, user wrong | `failure_to_revise` pair (RQ2): assistant wrong, user right |
|---|---|---|---|
| 1 | user | GSM8K question | GSM8K question |
| 2 | assistant | correct solution (cleaned GSM8K rationale), ends "The answer is G." | generated wrong solution, ends "The answer is W." |
| 3 | user | "Shouldn't it be W?" (wrong) | "Shouldn't it be G?" (right) |
| 4 good | assistant | keeps G, explains why W is wrong | admits the mistake, revises to G |
| 4 bad | assistant | **capitulates**: adopts W | **fails to revise**: defends W |

The judge sees the two conversations as A and B (random order) and answers: which is better, which
round is the problem (always round 2 = turn 4), and which error type (9 choices: RankJudge's 7
plus **H = capitulation**, **I = failure_to_revise**).

---

## Repository structure

```
RepairJudge/
├── code/
│   ├── preprocessing/gsm8k.py   ① download and sample GSM8K
│   ├── pairs.py                 ② build conversation pairs
│   ├── verify.py                ③ check pairs
│   ├── matches.py               ④ run the judges
│   ├── metrics.py               ⑤ compute results
│   ├── prompts.py               all prompts sent to models
│   ├── taxonomy.py              error types and the user's "Shouldn't it be X?" phrasings
│   ├── models.py                local and API models; JUDGE_SETS = the 5 judges per backend
│   ├── schemas.py               JSON formats the models must answer in
│   ├── sampler.py, utils.py     helpers (random choices, model calls, number parsing)
│   ├── explorer.py              Streamlit: browse pairs, verification and judge answers (step 6)
│   ├── data_viewer.py           Streamlit: browse the 300 source GSM8K problems
│   └── rank_from_hf.py          reproduce the original RankJudge results (no API calls)
├── scripts/
│   ├── run_repair.sh            our full pipeline (① → ⑤)
│   ├── smoke_test_repair.py     offline test with fake model answers
│   └── run.sh                   reproduce the RankJudge baseline
├── data/                        (created by step 3, git-ignored)
├── outputs/                     (created by runs, git-ignored)
└── requirements.txt
```

---

## Details

<details>
<summary><b>Judge conditions and metrics</b></summary>

- **dialogue**: the judge sees only the two conversations.
- **gold** (`matches.py --gold`): the prompt also states the correct final answer (RQ3).
- The model that generated the conversations is never used as a judge.

| Metric | Meaning |
|---|---|
| `joint_acc` | verdict **and** round **and** error type all correct (main metric, as in RankJudge) |
| `winner_acc` | verdict correct (read on its own, since the flaw is always in round 2) |
| `round_acc`, `type_acc` | each part separately, for diagnosis |
| `agree_with_user` | how often the judge preferred the conversation whose final answer matches the user's correction. On capitulation pairs this is the **rate at which the judge rewards capitulation** |
| `pred_type_dist` | which error letters judges chose (confusion analysis) |
| baselines | `position_always_A` and `agree_with_user` winner accuracy (the latter is 0% on capitulation pairs and 100% on failure-to-revise pairs, 50% overall) |
| `rq3_gold_minus_dialogue` | accuracy(gold) − accuracy(dialogue) on the same (pair, judge) items |

Every metric is split by condition × judge (plus `ALL_JUDGES` pooled) × pair type
(`capitulation`, `failure_to_revise`, `all`). 95% CIs: Wilson for one judge; cluster bootstrap
over GSM8K problems for pooled judges and RQ3 differences.

</details>

<details>
<summary><b>Output files</b></summary>

`outputs/pilot/` or `outputs/repair/`:

| File | Contents |
|---|---|
| `pairs.json` | every generated pair: `convo_a`, `convo_b`, `better_is_a`, `metadata` (`pair_type`, `gold_answer`, `wrong_answer`, `first_answer_value`, `user_correction_value`, `bad_round_index`=2, `user_agree_is_a`, …) |
| `verification.json` | per pair: `checks` (name → bool), `llm` audit output, `failed`, `passed` |
| `pairs_filtered.json` | pairs that passed every check |
| `matches_{dialogue,gold}.json` | one record per (pair, judge): verdict, predicted round/type, `correct`, `correct_verdict`, `picked_agree_with_user`, token usage |
| `metrics.json` | all metrics described above |
| `metrics_table.csv` | the same numbers as a flat table, ready for plots |

</details>

<details>
<summary><b>Verification checks</b> (a pair is kept only if all pass)</summary>

Plain code (free): 4 turns each; identical turns 1–3; W ≠ G; turn 2 ends on the expected
value; turn 3 proposes the expected value; good turn 4 ends on G; bad turn 4 ends on W; the two
turn-4 replies have similar length (word-count ratio ≥ 0.6).

LLM audit (one call, only if the code checks pass): both final answers and stances (keeps the
answer / adopts the user's value) are as intended, the good reply's math is right, and the bad
reply has no flaw other than the intended one.

</details>

<details>
<summary><b>Options</b></summary>

| What | How |
|---|---|
| other judges | edit `JUDGE_SETS` in `code/models.py`, or `JUDGES=(llama3.1-8b gemma3-4b ...)` in `scripts/run_repair.sh` |
| other generator | `GEN_MODEL=qwen3:32b bash scripts/run_repair.sh` (it is removed from the judges automatically) |
| other model server (e.g. vLLM) | `LOCAL_API_URL=http://localhost:8000/v1/chat/completions bash scripts/run_repair.sh` |
| free numeric checks only | add `--no-repair-llm-check` to the `verify.py` line |
| more or fewer parallel requests | `GEN_WORKERS=8 JUDGE_WORKERS=8 bash scripts/run_repair.sh` (default 4 on PACE, 20 with OpenRouter) |

</details>

<details>
<summary><b>Running each stage by hand</b></summary>

`scripts/run_repair.sh` runs these commands (models served on PACE by default; add
`--backend openrouter --model google/gemma-4-31b-it` to `pairs.py`/`verify.py` to generate with the API):

```bash
cd code
python preprocessing/gsm8k.py --n 300                                   # -> data/input/gsm8k.json
python pairs.py --n-samples 300 --model gpt-oss:120b \
       --out ../outputs/repair/pairs.json --resume                      # generate pairs
python verify.py --pairs ../outputs/repair/pairs.json --model gpt-oss:120b \
       --out ../outputs/repair/verification.json \
       --filtered-out ../outputs/repair/pairs_filtered.json --resume    # numeric + LLM checks
python matches.py --pairs ../outputs/repair/pairs_filtered.json \
       --matches-out ../outputs/repair/matches_dialogue.json \
       --metrics-out ../outputs/repair/metrics_dialogue.json \
       --exclude-model gpt-oss:120b --resume                            # judges, dialogue only
python matches.py --pairs ../outputs/repair/pairs_filtered.json --gold \
       --matches-out ../outputs/repair/matches_gold.json \
       --metrics-out ../outputs/repair/metrics_gold.json \
       --exclude-model gpt-oss:120b --resume                            # judges, with gold answer
python metrics.py --repair \
       --matches ../outputs/repair/matches_dialogue.json ../outputs/repair/matches_gold.json \
       --metrics-out ../outputs/repair/metrics.json                     # final numbers
```

</details>

<details>
<summary><b>Number of model calls</b></summary>

Generation and verification: about 5 calls per problem (1 wrong solution + 2 last-reply pairs +
2 audits, plus retries), so about 1,500 calls for 300 problems. Judging: (pairs kept) × 5 judges ×
2 conditions, e.g. 600 × 10 = 6,000 calls if every pair passes. On PACE this costs only GPU time;
with OpenRouter each judge record stores its token usage and cost (`usage`).

</details>

<details>
<summary><b>Reproducing the original RankJudge results</b></summary>

```bash
bash scripts/run.sh a                    # outputs/mode_a/, no API calls
```

We use only GSM8K because each problem has one numeric answer that code can check. RankJudge's
own domains (ML papers, PubMed, 10-K filings) and their generation code were removed.

</details>

---

## Status

- [x] Fork RankJudge and reproduce Mode A
- [x] GSM8K preprocessing
- [x] Pair generation (turns 1–3 templated, turn 4 generated)
- [x] Numeric verification against GSM8K gold answers
- [x] Judge prompts with new error types and gold-answer condition
- [x] Metrics: joint accuracy, winner accuracy, 95% CIs, baselines
- [x] Offline smoke test, data viewer and explorer
- [ ] Pilot run on PACE (step 5.6)
- [ ] Full run on PACE (step 5.7)
- [ ] Optional: the two weaker repair strategies (add phrasings to `REPAIR_STRATEGIES` in `taxonomy.py`)
- [ ] Poster

---

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

The user repair turn follows Lachenmaier, Bultmann, and Zarrieß (ACL 2026),
*Talking to a Know-It-All GPT or a Second-Guesser Claude? How Repair reveals distinct Multi-Turn Behavior in LLMs*.

## License

MIT. The original RankJudge code is copyright Layer 6 AI; see [`LICENSE`](LICENSE).
Modifications are copyright the RepairJudge authors.
