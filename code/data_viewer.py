"""Simple Streamlit viewer for our data.

Pick a dataset and an item in the sidebar; the main page shows that item.
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
# data/input/ exists after running preprocessing; outputs/repair/ after unzipping the shared dataset
DEFAULT_PROBLEMS = next(
    (p for p in (os.path.join(ROOT, "data", "input", "gsm8k.json"),
                 os.path.join(ROOT, "outputs", "repair", "gsm8k.json")) if os.path.exists(p)),
    os.path.join(ROOT, "data", "input", "gsm8k.json"))
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


def search_box(label):
    return st.sidebar.text_input(label).strip().lower()


# ------------------------------------------------------------ GSM8K page

def page_problems():
    path = st.sidebar.text_input("Problems file", DEFAULT_PROBLEMS)
    problems = read(path)
    if problems is None:
        st.warning(f"`{path}` not found. Unzip the shared dataset into `outputs/` "
                   "(README step 3), or create it with:\n\n"
                   "```bash\ncd code && python preprocessing/gsm8k.py --n 300\n```")
        return

    q = search_box("Search question text or id")
    view = [p for p in problems
            if not q or q in p["id"].lower() or q in p["question"].lower()]
    st.sidebar.caption(f"{len(view)} of {len(problems)} problems")
    if not view:
        st.info("No problems match the search.")
        return
    by_id = {p["id"]: p for p in view}
    p = by_id[st.sidebar.selectbox("Open a problem", list(by_id))]

    st.header(p["id"])
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

    types = st.sidebar.multiselect("Pair type", ["capitulation", "failure_to_revise"],
                                   default=["capitulation", "failure_to_revise"])
    status = st.sidebar.radio("Verification", ["all", "passed", "failed"], horizontal=True) \
        if verif else "all"
    q = search_box("Search pair or problem id")

    def keep(p):
        m = p["metadata"]
        if m["pair_type"] not in types:
            return False
        if q and q not in p["id"].lower() and q not in str(m["problem_id"]).lower():
            return False
        if status != "all":
            v = verif.get(p["id"])
            return bool(v and v["passed"]) == (status == "passed")
        return True

    by_id = {p["id"]: p for p in pairs if keep(p)}
    st.sidebar.caption(f"{len(by_id)} of {len(pairs)} pairs")
    if not by_id:
        st.info("No pairs match the filters.")
        return
    sel = st.sidebar.selectbox(
        "Open a pair", list(by_id),
        format_func=lambda i: f"{i}  ({by_id[i]['metadata']['pair_type']}, "
                              f"{by_id[i]['metadata']['problem_id']})")
    pair = by_id[sel]
    m = pair["metadata"]
    good, bad = good_bad(pair)

    flaw = "capitulation" if m["pair_type"] == "capitulation" else "failure to revise"
    st.header(sel)
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
    page = st.sidebar.radio("Dataset", ["GSM8K problems", "Generated pairs"])
    st.sidebar.divider()
    if page == "GSM8K problems":
        page_problems()
    else:
        page_pairs()


main()
