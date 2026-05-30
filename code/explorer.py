"""Streamlit per-pair explorer for RankJudge.

Two modes for the two ways the data can live:

  Mode A (default) -- load the released Layer6/RankJudge dataset from
                      Hugging Face (`pairs` + `matches` configs). The
                      published slice is already filtered, so there is
                      no separate "all" view and no verification record.

  Mode B           -- load the full local pipeline output from a from-scratch
                      run (`outputs/mode_b/`): pairs.json, pairs_filtered.json,
                      verification.json, matches.json.

Run from `code/`:

  streamlit run explorer.py                          # Mode A
  streamlit run explorer.py -- --mode b              # Mode B (from-scratch run)
  streamlit run explorer.py -- --mode b --dir ../outputs/mode_b

Requires `streamlit` in addition to the base deps.
"""

import argparse, json, os, re, sys

import streamlit as st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["a", "b"], default="a",
                   help="a: released HF dataset (default); b: local pipeline outputs")
    p.add_argument("--dir", default="../outputs/mode_b",
                   help="local pipeline dir (Mode B inputs)")
    p.add_argument("--repo", default="Layer6/RankJudge",
                   help="(Mode A) HF dataset repo id")
    p.add_argument("--split", default="train", help="(Mode A) HF split")
    p.add_argument("--cache-dir", default="../data",
                   help="(Mode A) HF datasets cache dir")
    p.add_argument("--init-elo", type=int, default=1500,
                   help="(Mode B) BT anchor for the top_removed cut")
    p.add_argument("--top-pct", type=float, default=0.05,
                   help="(Mode B) top fraction by pair Elo dropped for top_removed")
    return p.parse_args(sys.argv[1:])


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_top_removed_ids(matches, init_elo, top_pct):
    """Pair IDs that make up the `top_removed` evaluation slice -- exactly what
    metrics.py keeps: drop trivial/incomplete pairs, then drop the top --top-pct
    by BT pair-Elo. Returns empty on any failure (e.g. missing deps)."""
    from types import SimpleNamespace
    try:
        from metrics import _filter_trivial_pairs, _select_top_pids
        from utils import collect_matches
        args = SimpleNamespace(init_elo=init_elo, top_pct=top_pct)
        filtered_matches, _, _ = _filter_trivial_pairs(collect_matches(matches, "correct"))
        return {m[1] for m in filtered_matches} - _select_top_pids(filtered_matches, args)
    except Exception:
        return set()


@st.cache_data(show_spinner=False)
def load_local(data_dir, init_elo, top_pct):
    pairs = load_json(os.path.join(data_dir, "pairs.json"))
    pairs_filtered = load_json(os.path.join(data_dir, "pairs_filtered.json"))
    verification = load_json(os.path.join(data_dir, "verification.json"))
    matches = load_json(os.path.join(data_dir, "matches.json"))
    by_pair = {}
    for m in matches:
        by_pair.setdefault(m["id"], []).append(m)
    verif_by_id = {v["id"]: v for v in verification}
    filtered_ids = {p["id"] for p in pairs_filtered}
    top_removed_ids = compute_top_removed_ids(matches, init_elo, top_pct)
    return pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, top_removed_ids


@st.cache_data(show_spinner="Loading Layer6/RankJudge from Hugging Face...")
def load_hf(repo, split, cache_dir):
    from datasets import load_dataset
    ds_pairs = load_dataset(repo, "pairs", split=split, cache_dir=cache_dir)
    ds_matches = load_dataset(repo, "matches", split=split, cache_dir=cache_dir)
    pairs = [dict(r) for r in ds_pairs]
    matches = [dict(r) for r in ds_matches]
    # The published slice is already filtered. No verification records.
    pairs_filtered = pairs
    verif_by_id = {}
    by_pair = {}
    for m in matches:
        by_pair.setdefault(m["id"], []).append(m)
    filtered_ids = {p["id"] for p in pairs_filtered}
    # The released slice is already top-removed, so there is no further cut.
    return pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, set()


def parse_judge_raw(raw):
    if not raw or not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def verdict_letter(better_is_a):
    return "A" if better_is_a else "B"


def pair_signals(pair_id, matches_for_pair):
    def acc(ms, key):
        vals = [m["judge"].get(key) for m in ms if m["judge"].get(key) is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None
    preds = [m["judge"].get("answer") for m in matches_for_pair if m["judge"].get("answer")]
    split = (len(set(preds)) > 1) if preds else False
    final_acc = acc(matches_for_pair, "correct")
    return {
        "verdict_acc": acc(matches_for_pair, "correct_verdict"),
        "round_acc": acc(matches_for_pair, "correct_bad_round"),
        "behavior_acc": acc(matches_for_pair, "correct_behavior_type"),
        "final_acc": final_acc,
        "n": len(matches_for_pair),
        "split": split,
        "suspicious": (final_acc is not None and final_acc <= 0.25) or split,
    }


def format_round(convo, r):
    i = 2 * (r - 1)
    user = convo[i]["content"] if i < len(convo) else ""
    asst = convo[i + 1]["content"] if i + 1 < len(convo) else ""
    return user, asst


def render_round(convo, r, is_bad, flaw_type=None):
    user, asst = format_round(convo, r)
    flaw_tag = f"  <- INJECTED FLAW: {flaw_type}" if is_bad and flaw_type else ("  <- INJECTED FLAW" if is_bad else "")
    title = f"Round {r}{flaw_tag}"
    with st.expander(title, expanded=is_bad):
        if is_bad:
            msg = f"This is the round where the assistant is supposed to exhibit the flaw"
            if flaw_type:
                msg += f": **{flaw_type}**"
            st.warning(msg + ".")
        st.markdown("**USER**")
        st.write(user)
        st.markdown("**ASSISTANT**")
        st.write(asst)


def ok_icon(b):
    if b is True:
        return "PASS"
    if b is False:
        return "FAIL"
    return "-"


def apply_filters(pair_list, verif, by_pair, f, has_verif):
    def keep(p):
        meta = p["metadata"]
        if f["domain"] and p["domain"] not in f["domain"]:
            return False
        if f["asst"] and meta["assistant_behavior_type"] not in f["asst"]:
            return False
        if f["user"] and meta["user_behavior_type"] not in f["user"]:
            return False
        if f["nrounds"] and meta["n_rounds"] not in f["nrounds"]:
            return False
        if has_verif and f["verif_passed"] != "any":
            v = verif.get(p["id"])
            passed = bool(v and v["coherence"]["good_ok"] and v["coherence"]["bad_ok"]
                          and v["adherence"]["good_followed"] and v["adherence"]["bad_followed"]
                          and v["adherence"]["bad_flaw_round_correct"]
                          and (v["grounding"]["good"].get("grounded_rate") == 1.0)
                          and (v["grounding"]["bad"].get("grounded_rate") == 1.0))
            if f["verif_passed"] == "passed" and not passed:
                return False
            if f["verif_passed"] == "failed" and passed:
                return False
        return True
    result = [p for p in pair_list if keep(p)]
    sort = f["sort"]
    def sig_of(p):
        return pair_signals(p["id"], by_pair.get(p["id"], []))
    def key_acc(p, k):
        v = sig_of(p)[k]
        return (2 if v is None else v, p["id"])
    if sort == "id":
        result.sort(key=lambda p: p["id"])
    elif sort == "final_acc asc":
        result.sort(key=lambda p: key_acc(p, "final_acc"))
    elif sort == "verdict_acc asc":
        result.sort(key=lambda p: key_acc(p, "verdict_acc"))
    elif sort == "round_acc asc":
        result.sort(key=lambda p: key_acc(p, "round_acc"))
    elif sort == "behavior_acc asc":
        result.sort(key=lambda p: key_acc(p, "behavior_acc"))
    elif sort == "suspicious first":
        result.sort(key=lambda p: (not sig_of(p)["suspicious"], p["id"]))
    return result


def render_sidebar(mode, pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, top_removed_ids):
    has_verif = bool(verif_by_id)
    has_all_view = mode == "b"

    st.sidebar.header("Dataset")
    if has_all_view:
        pairs_final = [p for p in pairs_filtered if p["id"] in top_removed_ids]
        options = [f"all ({len(pairs)})",
                   f"filtered ({len(pairs_filtered)})",
                   f"final ({len(pairs_final)})"]
        ds = st.sidebar.radio("Pairs to browse", options, index=1)
        if ds.startswith("all"):
            pair_list, viewing_all = pairs, True
        elif ds.startswith("filtered"):
            pair_list, viewing_all = pairs_filtered, False
        else:  # final -- the published-style eval slice
            pair_list, viewing_all = pairs_final, False
            st.sidebar.caption("eval slice: trivial/incomplete + top-Elo pairs dropped")
    else:
        st.sidebar.caption(f"published slice ({len(pairs_filtered)} pairs)")
        pair_list = pairs_filtered
        viewing_all = False

    st.sidebar.header("Filters")
    domains = sorted({p["domain"] for p in pair_list})
    assts = sorted({p["metadata"]["assistant_behavior_type"] for p in pair_list})
    users = sorted({p["metadata"]["user_behavior_type"] for p in pair_list})
    nrs = sorted({p["metadata"]["n_rounds"] for p in pair_list})
    f = {
        "domain": st.sidebar.multiselect("domain", domains),
        "asst": st.sidebar.multiselect("assistant_behavior", assts),
        "user": st.sidebar.multiselect("user_behavior", users),
        "nrounds": st.sidebar.multiselect("n_rounds", nrs),
        "verif_passed": (st.sidebar.selectbox("verification passed",
                                              ["any", "passed", "failed"])
                         if (has_verif and viewing_all) else "any"),
        "sort": st.sidebar.selectbox("sort by",
                                     ["id", "final_acc asc", "verdict_acc asc",
                                      "round_acc asc", "behavior_acc asc", "suspicious first"]),
    }
    filtered = apply_filters(pair_list, verif_by_id, by_pair, f, has_verif)

    st.sidebar.header(f"Pairs ({len(filtered)})")
    if not filtered:
        st.sidebar.warning("No pairs match the filters.")
        return None

    qp_id = st.query_params.get("id")
    ids = [p["id"] for p in filtered]

    if "pair_id" not in st.session_state or st.session_state.pair_id not in ids:
        st.session_state.pair_id = qp_id if qp_id in ids else ids[0]

    def label_for(p):
        sig = pair_signals(p["id"], by_pair.get(p["id"], []))
        fa = sig["final_acc"]
        fa_s = f"{fa:.0%}" if fa is not None else "?"
        return f"{p['id']} | {p['domain']} | {p['metadata']['assistant_behavior_type']} | final={fa_s}"

    cols = st.sidebar.columns(3)
    if cols[0].button("< prev", use_container_width=True):
        cur = ids.index(st.session_state.pair_id)
        if cur > 0:
            st.session_state.pair_id = ids[cur - 1]
            st.rerun()
    if cols[1].button("next >", use_container_width=True):
        cur = ids.index(st.session_state.pair_id)
        if cur < len(ids) - 1:
            st.session_state.pair_id = ids[cur + 1]
            st.rerun()
    jump = cols[2].text_input("jump", label_visibility="collapsed", placeholder="id")
    if jump and jump in ids and jump != st.session_state.pair_id:
        st.session_state.pair_id = jump
        st.rerun()

    sel = st.sidebar.selectbox(
        "pair", ids, key="pair_id",
        format_func=lambda i: label_for(next(p for p in filtered if p["id"] == i)))

    st.query_params["id"] = sel

    return sel


def render_overview(pair, sig, verif):
    m = pair["metadata"]
    c1, c2, c3 = st.columns(3)
    c1.metric("domain", pair["domain"])
    c2.metric("n_rounds", m["n_rounds"])
    c3.metric("bad_round_index", m["bad_round_index"])
    c1.write(f"**id:** `{pair['id']}`")
    c1.write(f"**better_is_a:** {pair['better_is_a']}  (good = convo_{'a' if pair['better_is_a'] else 'b'})")
    c2.write(f"**assistant_behavior:** {m['assistant_behavior_type']}")
    c3.write(f"**user_behavior:** {m['user_behavior_type']}")

    st.divider()
    st.subheader(f"Judge signal across all judges (n={sig['n']})")
    g1, g2, g3, g4 = st.columns(4)
    def fmt(v):
        return "-" if v is None else f"{v:.0%}"
    g1.metric("final_correct", fmt(sig["final_acc"]))
    g2.metric("verdict_acc", fmt(sig["verdict_acc"]))
    g3.metric("bad_round_acc", fmt(sig["round_acc"]))
    g4.metric("behavior_acc", fmt(sig["behavior_acc"]))

    if verif:
        c, a = verif["coherence"], verif["adherence"]
        st.caption(f"verification: coherence(good={ok_icon(c['good_ok'])}, bad={ok_icon(c['bad_ok'])}) | "
                   f"adherence(good={ok_icon(a['good_followed'])}, bad={ok_icon(a['bad_followed'])}, flaw_round={ok_icon(a['bad_flaw_round_correct'])}) | "
                   f"grounding(good={verif['grounding']['good'].get('grounded_rate')}, bad={verif['grounding']['bad'].get('grounded_rate')})")

    with st.expander("Reference context (paper / source)", expanded=False):
        st.text(m.get("context", ""))


def render_md(text):
    """Render long text as markdown but keep single line breaks (the plans are
    one `Round N:` per line) and convert \\(..\\)/\\[..\\] LaTeX to $..$/$$..$$."""
    s = text if isinstance(text, str) else str(text)
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.S)        # inline LaTeX
    s = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", s, flags=re.S)      # block LaTeX
    s = s.replace("\n", "  \n")                                  # single \n -> hard break
    st.markdown(s)


def render_plan(pair):
    m = pair["metadata"]
    left, right = st.columns(2)
    with left:
        st.subheader("Good plan")
        st.caption(f"user_behavior: {m['user_behavior_type']}")
        render_md(pair["plan"]["good"])
    with right:
        st.subheader("Bad plan")
        st.caption(f"assistant_behavior: {m['assistant_behavior_type']}  |  bad_round_index: **{m['bad_round_index']}**")
        render_md(pair["plan"]["bad"])


def render_convos(pair):
    m = pair["metadata"]
    good_convo, bad_convo = ((pair["convo_a"], pair["convo_b"]) if pair["better_is_a"]
                             else (pair["convo_b"], pair["convo_a"]))
    bad_round = m["bad_round_index"]
    n = m["n_rounds"]
    left, right = st.columns(2)
    with left:
        st.subheader(f"Good convo  (= convo_{'a' if pair['better_is_a'] else 'b'})")
        for r in range(1, n + 1):
            render_round(good_convo, r, is_bad=False)
    with right:
        st.subheader(f"Bad convo  (= convo_{'b' if pair['better_is_a'] else 'a'})")
        for r in range(1, n + 1):
            render_round(bad_convo, r, is_bad=(r == bad_round),
                         flaw_type=m["assistant_behavior_type"])


def render_verification(verif):
    if not verif:
        st.info("No verification record for this pair.")
        return
    c, a, g = verif["coherence"], verif["adherence"], verif["grounding"]
    st.subheader("Coherence (is the plan internally consistent?)")
    st.write(f"good_ok: {ok_icon(c['good_ok'])}  |  bad_ok: {ok_icon(c['bad_ok'])}")
    if c.get("good_issue"):
        st.error(f"good_issue: {c['good_issue']}")
    if c.get("bad_issue"):
        st.error(f"bad_issue: {c['bad_issue']}")

    st.subheader("Adherence (did the convo follow the plan?)")
    st.write(f"good_followed: {ok_icon(a['good_followed'])}  |  bad_followed: {ok_icon(a['bad_followed'])}  |  "
             f"bad_flaw_round_correct: {ok_icon(a['bad_flaw_round_correct'])}")
    if a.get("good_issue"):
        st.error(f"good_issue: {a['good_issue']}")
    if a.get("bad_issue"):
        st.error(f"bad_issue: {a['bad_issue']}")

    st.subheader("Grounding (are assistant claims supported by the source?)")
    st.caption("The bad convo's flawed round is excluded from grounding (flaw may be intentionally ungrounded).")
    for label in ("good", "bad"):
        gg = g[label]
        rounds = gg.get("rounds", [])
        rate = gg.get("grounded_rate")
        total = sum(len(r["claims"]) for r in rounds)
        grounded = sum(1 for r in rounds for c in r["claims"] if c.get("grounded"))
        ungrounded = total - grounded
        rate_s = f"{rate:.0%}" if isinstance(rate, (int, float)) else str(rate)
        st.markdown(f"**{label} convo** - grounded_rate: `{rate_s}`  |  "
                    f"claims: {total}  |  grounded: {grounded}  |  ungrounded: {ungrounded}")
        if not rounds:
            st.caption("(no rounds scored)")
            continue
        for r in rounds:
            claims = r.get("claims", [])
            n_ok = sum(1 for c in claims if c.get("grounded"))
            n_bad = len(claims) - n_ok
            head = f"Round {r['round_index']}  -  {n_ok}/{len(claims)} grounded" + (f"  ({n_bad} ungrounded)" if n_bad else "")
            with st.expander(head, expanded=bool(n_bad)):
                for c in claims:
                    st.markdown(f"- {ok_icon(c.get('grounded'))}  {c['claim']}")


def render_judges(pair, matches_for_pair):
    if not matches_for_pair:
        st.info("No matches for this pair.")
        return
    sig = pair_signals(pair["id"], matches_for_pair)
    def fmt(v):
        return "-" if v is None else f"{v:.0%}"
    st.caption(f"n={sig['n']}  |  final_correct={fmt(sig['final_acc'])}  |  "
               f"verdict_acc={fmt(sig['verdict_acc'])}  |  "
               f"bad_round_acc={fmt(sig['round_acc'])}  |  "
               f"behavior_acc={fmt(sig['behavior_acc'])}")

    gt_verdict = verdict_letter(pair["better_is_a"])
    gt_round = pair["metadata"]["bad_round_index"]
    gt_beh = pair["metadata"]["assistant_behavior_type"]
    st.caption(f"GT: verdict={gt_verdict}, bad_round={gt_round}, behavior={gt_beh}")

    rows = sorted(matches_for_pair, key=lambda m: m["model"]["name"])
    for m in rows:
        j = m["judge"]
        head = (f"**{m['model']['name']}**  |  "
                f"final={ok_icon(j.get('correct'))}  |  "
                f"verdict={j.get('answer')} {ok_icon(j.get('correct_verdict'))}  |  "
                f"round={j.get('bad_round_pred')} {ok_icon(j.get('correct_bad_round'))}  |  "
                f"behavior={j.get('behavior_type_pred')} {ok_icon(j.get('correct_behavior_type'))}")
        with st.expander(head, expanded=False):
            parsed = parse_judge_raw(j.get("raw", ""))
            if parsed.get("analysis"):
                st.markdown("**analysis**")
                st.write(parsed["analysis"])
            else:
                st.markdown("**raw (unparsed)**")
                st.code(j.get("raw") or "", language="json")


def main():
    args = parse_args()

    st.set_page_config(page_title="RankJudge data explorer", layout="wide")
    st.title("RankJudge data explorer")

    if args.mode == "a":
        st.caption(f"Mode A -- released dataset `{args.repo}` (split: {args.split})")
        try:
            pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, top_removed_ids = load_hf(
                args.repo, args.split, os.path.abspath(args.cache_dir))
        except Exception as e:
            st.error(f"Failed to load `{args.repo}` from Hugging Face: {e}")
            return
    else:
        data_dir = os.path.abspath(args.dir)
        st.caption(f"Mode B -- local data dir: `{data_dir}`")
        try:
            pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, top_removed_ids = load_local(
                data_dir, args.init_elo, args.top_pct)
        except FileNotFoundError as e:
            st.error(f"Missing file in {data_dir}: {e}")
            return

    sel = render_sidebar(
        args.mode, pairs, pairs_filtered, verif_by_id, by_pair, filtered_ids, top_removed_ids)
    if sel is None:
        return

    pair = next((p for p in pairs if p["id"] == sel), None)
    if pair is None:
        st.error(f"Pair {sel} not found.")
        return
    verif = verif_by_id.get(sel)
    matches_for_pair = by_pair.get(sel, [])
    sig = pair_signals(sel, matches_for_pair)

    in_filtered = sel in filtered_ids
    badge = "[filtered-in]" if in_filtered else "[filtered-out]"
    if args.mode == "a":
        badge = "[published]"
    elif sel in top_removed_ids:
        badge += " [final]"
    st.markdown(f"### `{sel}` &nbsp; {badge}")

    tab_labels = ["Overview", "Plan", "Conversations"]
    if args.mode == "b":
        tab_labels.append("Verification")
    tab_labels.append(f"Judges ({len(matches_for_pair)})")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview(pair, sig, verif)
    with tabs[1]:
        render_plan(pair)
    with tabs[2]:
        render_convos(pair)
    if args.mode == "b":
        with tabs[3]:
            render_verification(verif)
        with tabs[4]:
            render_judges(pair, matches_for_pair)
    else:
        with tabs[3]:
            render_judges(pair, matches_for_pair)


if __name__ == "__main__":
    main()
