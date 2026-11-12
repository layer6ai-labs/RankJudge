"""Simple Streamlit viewer for our data.

Pages:
  GSM8K problems   -- the preprocessed 300 problems (data/input/gsm8k.json)
  Generated pairs  -- conversation pairs + verification results from a run
                      (outputs/pilot/ or outputs/repair/)

Run from the repo root:
  streamlit run code/data_viewer.py

For judge answers per pair, use explorer.py instead.
"""

import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBLEMS = os.path.join(ROOT, "data", "input", "gsm8k.json")
RUN_DIRS = ["outputs/pilot", "outputs/repair", "outputs/smoke"]


# ------------------------------------------------------------------ loading

@st.cache_data
def load_json(path, mtime):
    with open(path) as f:
        return json.load(f)


def read(path):
    if not os.path.exists(path):
        return None
    return load_json(path, os.path.getmtime(path))


def problems_frame(problems):
    rows = []
    for p in problems:
        gold = float(p["gold_answer"])
        rows.append({
            "id": p["id"],
            "gold_answer": p["gold_answer"],
            "steps": len(p["gold_rationale"].splitlines()),
            "question_words": len(p["question"].split()),
            "integer_answer": gold == int(gold),
            "question": p["question"],
        })
    return pd.DataFrame(rows)


def magnitude_bucket(x):
    x = abs(float(x))
    for limit, label in [(10, "< 10"), (100, "10-99"), (1000, "100-999"), (10000, "1k-9.9k")]:
        if x < limit:
            return label
    return ">= 10k"


# ------------------------------------------------------------ GSM8K page

def page_problems():
    st.header("GSM8K problems")
    path = st.sidebar.text_input("Problems file", DEFAULT_PROBLEMS)
    problems = read(path)
    if problems is None:
        st.warning(f"`{path}` not found. Create it with:\n\n"
                   "```bash\ncd code && python preprocessing/gsm8k.py --n 300\n```")
        return
    df = problems_frame(problems)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Problems", len(df))
    c2.metric("Median solution steps", int(df["steps"].median()))
    c3.metric("Median question length", f"{int(df['question_words'].median())} words")
    c4.metric("Integer answers", f"{df['integer_answer'].mean():.0%}")

    st.caption("Each problem becomes 2 pairs: a *wrong_correction* pair (tests capitulation) and a "
               "*right_correction* pair (tests failure to revise). The cleaned rationale below is "
               "the assistant's correct first answer (turn 2) in the wrong_correction pair.")

    left, right = st.columns(2)
    with left:
        st.subheader("Solution steps")
        st.bar_chart(df["steps"].value_counts().sort_index(), x_label="steps", y_label="problems")
    with right:
        st.subheader("Size of the gold answer")
        order = ["< 10", "10-99", "100-999", "1k-9.9k", ">= 10k"]
        buckets = df["gold_answer"].map(magnitude_bucket).value_counts().reindex(order, fill_value=0)
        st.bar_chart(buckets, x_label="gold answer", y_label="problems")

    st.subheader("Browse")
    f1, f2 = st.columns([2, 1])
    query = f1.text_input("Search question text or id")
    lo, hi = int(df["steps"].min()), int(df["steps"].max())
    step_range = f2.slider("Solution steps", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
    view = df[df["steps"].between(*step_range)]
    if query:
        q = query.lower()
        view = view[view["question"].str.lower().str.contains(q, regex=False) | view["id"].str.contains(q, regex=False)]
    st.dataframe(view[["id", "gold_answer", "steps", "question_words", "question"]],
                 hide_index=True, width="stretch", height=280)
    st.caption(f"{len(view)} of {len(df)} problems shown")

    if view.empty:
        return
    pid = st.selectbox("Open a problem", view["id"].tolist())
    p = next(x for x in problems if x["id"] == pid)
    st.markdown(f"**Question**\n\n{p['question']}")
    st.markdown(f"**Gold answer:** `{p['gold_answer']}`")
    st.markdown("**Cleaned rationale** (used as the assistant's correct answer)")
    st.code(p["gold_rationale"], language=None)
    with st.expander("Original GSM8K solution (with <<calculator>> annotations)"):
        st.code(p["gold_solution"], language=None)


# ------------------------------------------------------------ pairs page

def good_bad(pair):
    return ((pair["convo_a"], pair["convo_b"]) if pair["better_is_a"]
            else (pair["convo_b"], pair["convo_a"]))


def page_pairs():
    st.header("Generated pairs")
    existing = [d for d in RUN_DIRS if os.path.exists(os.path.join(ROOT, d, "pairs.json"))]
    choice = st.sidebar.selectbox("Run folder", existing + ["(other)"]) if existing else "(other)"
    run_dir = st.sidebar.text_input("Path", os.path.join(ROOT, "outputs/pilot")) if choice == "(other)" \
        else os.path.join(ROOT, choice)

    pairs = read(os.path.join(run_dir, "pairs.json"))
    if pairs is None:
        st.warning(f"No `pairs.json` in `{run_dir}`. Run the pipeline first, e.g.\n\n"
                   "```bash\nN_PROBLEMS=10 OUT=../outputs/pilot bash scripts/run_repair.sh\n```")
        return
    verif = {v["id"]: v for v in (read(os.path.join(run_dir, "verification.json")) or [])}

    rows = []
    for p in pairs:
        m, v = p["metadata"], verif.get(p["id"])
        rows.append({"id": p["id"], "problem_id": m["problem_id"], "pair_type": m["pair_type"],
                     "gold": m["gold_answer"], "wrong": m["wrong_answer"],
                     "user_says": m["user_correction_value"],
                     "verified": None if v is None else v["passed"],
                     "failed_checks": "" if v is None else ", ".join(v["failed"])})
    df = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs", len(df))
    c2.metric("wrong_correction (capitulation)", int((df["pair_type"] == "wrong_correction").sum()))
    c3.metric("right_correction (failure to revise)", int((df["pair_type"] == "right_correction").sum()))
    if verif:
        c4.metric("Passed verification", f"{int(df['verified'].fillna(False).sum())} / {len(df)}")
    else:
        c4.metric("Passed verification", "not run")

    if verif:
        failed = pd.Series([c for v in verif.values() for c in v["failed"]], dtype=str)
        if not failed.empty:
            st.subheader("Why pairs failed verification")
            st.bar_chart(failed.value_counts(), x_label="check", y_label="pairs")

    st.subheader("Browse")
    f1, f2 = st.columns(2)
    types = f1.multiselect("Pair type", ["wrong_correction", "right_correction"],
                           default=["wrong_correction", "right_correction"])
    status = f2.radio("Verification", ["all", "passed", "failed"], horizontal=True)
    view = df[df["pair_type"].isin(types)]
    if status != "all":
        view = view[view["verified"].fillna(False) == (status == "passed")]
    st.dataframe(view, hide_index=True, width="stretch", height=250)
    if view.empty:
        return

    sel = st.selectbox("Open a pair", view["id"].tolist(),
                       format_func=lambda i: f"{i}  ({df.set_index('id').loc[i, 'pair_type']}, "
                                             f"{df.set_index('id').loc[i, 'problem_id']})")
    pair = next(p for p in pairs if p["id"] == sel)
    m = pair["metadata"]
    good, bad = good_bad(pair)

    flaw = "capitulation" if m["pair_type"] == "wrong_correction" else "failure to revise"
    st.markdown(f"**{m['pair_type']}**: tests **{flaw}**. Gold answer `{m['gold_answer']}`, "
                f"assistant first said `{m['first_answer_value']}`, user suggests `{m['user_correction_value']}`. "
                f"Good final answer `{m['good_final_value']}`, bad final answer `{m['bad_final_value']}`. "
                f"Shown to judges with the good conversation as **{'A' if pair['better_is_a'] else 'B'}**.")

    st.markdown("##### Turns 1-3 (same in both conversations)")
    for i, t in enumerate(good[:3], 1):
        with st.chat_message("user" if t["role"] == "user" else "assistant"):
            st.markdown(f"*turn {i}*\n\n{t['content']}")

    st.markdown("##### Turn 4 (the only difference)")
    left, right = st.columns(2)
    with left:
        st.success("Good response")
        st.markdown(good[3]["content"])
        st.caption(f"{len(good[3]['content'].split())} words")
    with right:
        st.error(f"Bad response ({flaw})")
        st.markdown(bad[3]["content"])
        st.caption(f"{len(bad[3]['content'].split())} words")

    v = verif.get(sel)
    if v:
        with st.expander(f"Verification: {'PASS' if v['passed'] else 'FAIL'}", expanded=not v["passed"]):
            st.dataframe(pd.DataFrame([{"check": k, "ok": ok} for k, ok in v["checks"].items()]),
                         hide_index=True)
            if v.get("llm"):
                st.json(v["llm"])
    if m.get("wrong_solution_mistake"):
        st.caption(f"Mistake injected into the wrong solution: {m['wrong_solution_mistake']}")


# ------------------------------------------------------------------ main

def main():
    st.set_page_config(page_title="RepairJudge data viewer", layout="wide")
    st.title("RepairJudge data viewer")
    page = st.sidebar.radio("Page", ["GSM8K problems", "Generated pairs"])
    if page == "GSM8K problems":
        page_problems()
    else:
        page_pairs()


main()
