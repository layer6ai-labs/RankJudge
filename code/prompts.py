"""All prompt templates for the benchmark suite."""

from taxonomy import ASSISTANT_BEHAVIORS

# --- Conversation generation (pairs.py) ---

GOOD_CONVO = """This conversation will be used in a pairwise comparison against a counterpart conversation that contains a subtle "{flaw_name}" flaw in exactly one round. The goal of the pair is to test whether an LLM judge can pick the better conversation — which only works if your conversation is absolutely correct on every dimension and naturally embodies the paired virtue, without performing it loudly. If your conversation is obviously stronger than a clean conversation with one subtle flaw, the test becomes trivial. Aim to read as ordinary high quality, not as a demonstration of the virtue.

Given this reference document, generate a natural multi-turn conversation ({n} rounds, each round = 1 user + 1 assistant message).

The context is reference material - it defines what the assistant knows about the topic. The conversation does NOT need to cover all of it. A focused discussion exploring a few topics in depth is better than superficially touching everything.

First, write a turn-by-turn plan with one labeled entry per round in the form `Round 1: ...`, `Round 2: ...`, ..., `Round {n}: ...`. Every round MUST have its own entry — do not collapse rounds into a single paragraph or skip ahead. For each round, capture (a) what the user asks and what motivates it (building on the prior round where natural), and (b) what the assistant's response will focus on from the reference material. The plan should form a coherent arc, not isolated Q&A. Then write the conversation faithfully executing the plan.

User behavior: "{user_behavior_name}" - {user_behavior_desc}

Assistant virtue to embody naturally (the quality every round must quietly live up to — do NOT overact it): {assistant_virtue}

Paired flaw the counterpart conversation will contain (for your awareness only — do NOT mention, contrast, or over-compensate against it; just never exhibit it): {assistant_flaw}

User rules:
- Has read the reference material. Asks questions driven by genuine curiosity, not to quiz the assistant.
- Apply the above behavior consistently across ALL rounds - it should shape what the user asks, how they react, and how the conversation progresses.
- NOT a passive question-asker - reacts to answers, builds on them, pushes back, or redirects.
- Keep messages SHORT and natural (1-3 sentences). Real users don't write essays.

Assistant rules:
- Every round must be clean on every dimension: accurate, grounded, specific, on-topic, well-structured, responsive, and appropriately calibrated. No flaw category should surface in any round.
- Answers must be grounded in the provided context. Do NOT fabricate information beyond what the context provides.
- If a user question goes beyond what the context covers, honestly acknowledge the limitation rather than guessing or fabricating.
- Be specific - use the actual numbers, details, and conclusions from the reference material.
- Be concise and direct (2-5 sentences per response). Avoid filler, repetition, and preamble. Match the tone, length, and confidence of a typical clean response — do not inflate quality signals to appear more virtuous.

=== Reference Context ===
{context}"""

WORSE_CONVO = """Given this reference document, generate a natural multi-turn conversation ({n} rounds, each round = 1 user + 1 assistant message) where the chatbot has a specific weakness.

Chatbot weakness: **{behavior_type}**
Description: {behavior_desc}

The context is reference material - it defines what the assistant knows about the topic. The conversation does NOT need to cover all of it. A focused discussion is fine.

First, write a turn-by-turn plan in the "reasoning" field with one labeled entry per round in the form `Round 1: ...`, `Round 2: ...`, ..., `Round {n}: ...`. Every round MUST have its own entry — do not collapse the plan into a single justification paragraph or skip ahead. For each round, capture (a) what the user asks and what motivates it (reacting to the prior round where natural), and (b) what the assistant's response will focus on from the reference material. The plan should form a coherent arc, not isolated Q&A. Pick one round to be the flawed one and, inside that round's entry, describe concretely how the "{behavior_type}" weakness will surface there — the clean rounds before and after should carry the conversation as if no flaw exists. Set "bad_round_index" to that round (1-indexed). Then write the conversation faithfully executing the plan.

User behavior: "{user_behavior_name}" - {user_behavior_desc}

User rules:
- Has read the reference material. Asks questions driven by genuine curiosity.
- Apply the above behavior consistently across ALL rounds.
- NOT a passive question-asker - reacts to answers naturally. If the chatbot gives an evasive answer, the user may follow up, push back, or move on - whatever fits. Do NOT have the user explicitly diagnose or name the flaw.
- Keep messages SHORT and natural (1-3 sentences).

Chatbot (assistant) rules:
- Answers must be grounded in the provided context. Do NOT fabricate information beyond what the context provides.
- Be specific - use the actual numbers, details, and conclusions from the reference material.
- Be concise and direct (2-5 sentences per response). Avoid filler, repetition, and preamble.
- In exactly 1 round, the assistant response should exhibit the "{behavior_type}" weakness. The flaw should require careful reading to detect - a reader skimming should NOT notice anything wrong. Keep the same confident tone and similar response length as other rounds.
- The flawed round must exhibit ONLY the "{behavior_type}" flaw - no other weakness category should appear in it. Every dimension of the response other than the named flaw must remain clean: accurate, grounded, specific, on-topic, well-structured, and responsive. All non-flawed rounds must be clean on every dimension.
- Do NOT mention or acknowledge that responses are flawed.

=== Reference Context ===
{context}"""

# --- Judge evaluation (matches.py) ---

BEHAVIOR_TO_LETTER = {name: chr(ord('A') + i) for i, name in enumerate(ASSISTANT_BEHAVIORS)}
LETTER_TO_BEHAVIOR = {v: k for k, v in BEHAVIOR_TO_LETTER.items()}

_BEHAVIOR_LIST = "\n".join(f"- **{BEHAVIOR_TO_LETTER[name]} ({name})**: {info['flaw']}" for name, info in ASSISTANT_BEHAVIORS.items())

JUDGE = """You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation's assistant provides better overall quality.
{context_section}
=== Conversation A ===
{text_a}

=== Conversation B ===
{text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories:
""" + _BEHAVIOR_LIST + """

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Respond in JSON with these fields in order:
- "analysis": your reasoning about which conversation is better and why
- "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- "problem_type": the letter (A-G) for the flaw category from the list above
- "verdict": "A" or "B" for the better conversation"""


# --- Verification (verify.py) ---

COHERENCE = """You are auditing a synthetic-data recipe before any conversation has been rendered. Given a reference document and a sampled plan, decide whether the plan can plausibly produce a natural multi-turn conversation.

A plan is INCOHERENT only when executing the stated virtue/flaw is semantically at odds with the user style for THIS specific reference material - not merely unusual. Surface-level mismatches (e.g. a skeptical user asking precise questions) are fine. Flag only genuine conflicts, e.g.:
- the flaw requires conditions the user style will never create (a "focused" user asking tightly scoped questions cannot surface a "no_clarification" flaw)
- the reference material lacks the content needed for the flaw to appear naturally (e.g. "factual_error" flaw but the material is too thin to have subtle facts to misstate)
- the bad_round_index sits outside [1, n_rounds] or collides with where the plan claims the conversation is just getting started / wrapping up

User behavior: "{user_behavior_name}" - {user_behavior_desc}
Assistant virtue (good convo): {virtue}
Assistant flaw (bad convo, "{assistant_behavior_name}"): {flaw}
n_rounds: {n_rounds}
bad_round_index (1-indexed): {bad_round_index}

=== Good plan (from generator) ===
{good_plan}

=== Bad plan (from generator) ===
{bad_plan}

=== Reference Context ===
{context}

Respond in JSON: {{"good_ok", "good_issue", "bad_ok", "bad_issue"}}. Leave *_issue as "" when ok. Keep issues to one sentence."""


ADHERENCE = """You are auditing whether two rendered conversations actually followed their generation plans. Read the plans and then the conversations and answer yes/no globally for each side.

For the GOOD conversation:
- Did the assistant consistently display the stated virtue across ALL rounds?
- Did the user maintain the stated user behavior across ALL rounds?

For the BAD conversation:
- Did the user maintain the stated user behavior?
- Does round {bad_round_index} (1-indexed) actually exhibit the "{assistant_behavior_name}" flaw described below?
- Is that the ONLY flawed assistant round? (If another round is clearly flawed, bad_flaw_round_correct is false.)

User behavior: "{user_behavior_name}" - {user_behavior_desc}
Assistant virtue: {virtue}
Assistant flaw "{assistant_behavior_name}": {flaw}
n_rounds: {n_rounds}

=== Good plan ===
{good_plan}

=== Good conversation ===
{good_convo}

=== Bad plan ===
{bad_plan}

=== Bad conversation (declared bad_round_index = {bad_round_index}) ===
{bad_convo}

Respond in JSON: {{"good_followed", "good_issue", "bad_followed", "bad_flaw_round_correct", "bad_issue"}}. Leave *_issue as "" when ok. One sentence per issue."""


GROUNDING = """You are fact-checking an assistant's responses against a reference document. For each listed assistant turn, extract every factual atomic claim it makes (specific numbers, named entities, mechanisms, cause/effect statements, attributions) and mark each claim as grounded or not.

Rules:
- A claim is `grounded: true` only if it is directly supported by the reference context below.
- Paraphrases and reasonable summaries of supported content count as grounded.
- Reasonable inferences or synthesis across multiple context statements also count as grounded, even if no single sentence states them verbatim.
- Unsupported extrapolation, invented details, or claims beyond the context are `grounded: false`.
- Ignore conversational filler, hedges, and meta-statements ("good question", "let me explain").
- Do NOT extract negative/absence claims ("X is not Y", "X doesn't mention Z") unless the context directly contradicts them - absence is not verifiable at claim granularity.
- Do NOT extract meta-statements where the assistant is evaluating another statement rather than asserting a fact ("it is inaccurate to claim...", "that earlier answer was wrong").
- Aim for 1-5 atomic claims per turn. Skip turns whose round_index is in skip_rounds: {skip_rounds}.

=== Assistant turns ({label}) ===
{turns}

=== Reference Context ===
{context}

Respond in JSON: {{"rounds": [{{"round_index", "claims": [{{"claim", "grounded"}}]}}]}}. Only include rounds you actually analyzed (exclude skip_rounds)."""
