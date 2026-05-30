"""JSON schemas for structured LLM outputs."""

from taxonomy import ASSISTANT_BEHAVIORS

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": ["user", "assistant"]},
        "content": {"type": "string"}
    },
    "required": ["role", "content"],
    "additionalProperties": False
}

GOOD_CONVO_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "string",
            "description": "Turn-by-turn plan with one labeled entry per round (`Round 1: ...`, `Round 2: ...`, ...). Each entry sketches the user's question and what the assistant's response will focus on. Every round must have its own entry."
        },
        "conversation": {
            "type": "array",
            "items": TURN_SCHEMA
        }
    },
    "required": ["plan", "conversation"],
    "additionalProperties": False
}


WORSE_CONVO_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "object",
            "description": "Plan which aspects of the material to discuss, what drives each round, how the dialogue evolves, and which rounds exhibit the weakness.",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Turn-by-turn plan with one labeled entry per round (`Round 1: ...`, `Round 2: ...`, ...). Each entry sketches the user's question and what the assistant's response will focus on; the entry for the flawed round must concretely describe how the weakness surfaces. Every round must have its own entry — do not write a single justification paragraph."
                },
                "bad_round_index": {
                    "type": "integer",
                    "description": "1-indexed round number where the assistant response exhibits the weakness. E.g. 2 means the assistant response in round 2 is flawed."
                }
            },
            "required": ["reasoning", "bad_round_index"],
            "additionalProperties": False
        },
        "conversation": {
            "type": "array",
            "items": TURN_SCHEMA
        }
    },
    "required": ["plan", "conversation"],
    "additionalProperties": False
}


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "Your reasoning about which conversation is better and why."
        },
        "worst_round": {
            "type": "integer",
            "description": "1-indexed round number of the worst assistant response in the conversation you did NOT pick."
        },
        "problem_type": {
            "type": "string",
            "enum": [chr(ord('A') + i) for i in range(len(ASSISTANT_BEHAVIORS))],
            "description": "The letter (A-I) for the flaw category of the worst round."
        },
        "verdict": {
            "type": "string",
            "enum": ["A", "B"],
            "description": "The better conversation: A or B."
        },
    },
    "required": ["analysis", "worst_round", "problem_type", "verdict"],
    "additionalProperties": False
}


# --- Verification (verify.py) ---

COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "good_ok": {"type": "boolean", "description": "True if the good plan is coherent for this context."},
        "good_issue": {"type": "string", "description": "Short reason if good_ok is false, empty otherwise."},
        "bad_ok": {"type": "boolean", "description": "True if the bad plan (including bad_round_index placement) is coherent."},
        "bad_issue": {"type": "string", "description": "Short reason if bad_ok is false, empty otherwise."},
    },
    "required": ["good_ok", "good_issue", "bad_ok", "bad_issue"],
    "additionalProperties": False,
}


ADHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "good_followed": {"type": "boolean", "description": "Good conversation consistently displays the stated virtue and user behavior."},
        "good_issue": {"type": "string", "description": "Short reason if good_followed is false, empty otherwise."},
        "bad_followed": {"type": "boolean", "description": "Bad conversation displays the stated flaw and user behavior."},
        "bad_flaw_round_correct": {"type": "boolean", "description": "The flaw lands at the declared bad_round_index and no other round is flawed."},
        "bad_issue": {"type": "string", "description": "Short reason if bad_followed or bad_flaw_round_correct is false, empty otherwise."},
    },
    "required": ["good_followed", "good_issue", "bad_followed", "bad_flaw_round_correct", "bad_issue"],
    "additionalProperties": False,
}


GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "rounds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "round_index": {"type": "integer", "description": "1-indexed round number."},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string", "description": "Atomic factual claim extracted from the assistant turn."},
                                "grounded": {"type": "boolean", "description": "True if directly supported by the reference context."},
                            },
                            "required": ["claim", "grounded"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["round_index", "claims"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rounds"],
    "additionalProperties": False,
}