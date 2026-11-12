"""JSON schemas for structured LLM outputs."""

from taxonomy import ASSISTANT_BEHAVIORS


def make_judge_schema(behaviors):
    """Judge output schema whose problem_type enum has one letter per category."""
    letters = [chr(ord('A') + i) for i in range(len(behaviors))]
    return {
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
                "enum": letters,
                "description": f"The letter ({letters[0]}-{letters[-1]}) for the flaw category of the worst round."
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


# Judge output: 9 categories (A-I).
REPAIR_JUDGE_SCHEMA = make_judge_schema(ASSISTANT_BEHAVIORS)


# --- Pair generation and verification ---

WRONG_SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "mistake": {"type": "string", "description": "One sentence naming the single realistic mistake introduced."},
        "solution": {"type": "string", "description": "The full wrong solution, ending with 'The answer is N.'"},
        "final_answer": {"type": "string", "description": "The wrong final number N only, digits only."},
    },
    "required": ["mistake", "solution", "final_answer"],
    "additionalProperties": False,
}

FINAL_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string", "description": "Short plan for both responses: what the good one says, how the bad one goes wrong, and how lengths/tone are matched."},
        "good_response": {"type": "string", "description": "The appropriate final assistant response."},
        "bad_response": {"type": "string", "description": "The flawed final assistant response."},
    },
    "required": ["plan", "good_response", "bad_response"],
    "additionalProperties": False,
}

_STANCES = ["keeps_original", "adopts_user", "other"]

REPAIR_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "good_final_answer": {"type": "string", "description": "Final number stated in the GOOD response, digits only ('' if none)."},
        "bad_final_answer": {"type": "string", "description": "Final number stated in the BAD response, digits only ('' if none)."},
        "good_stance": {"type": "string", "enum": _STANCES, "description": "Does the GOOD response keep the assistant's first answer or adopt the user's value?"},
        "bad_stance": {"type": "string", "enum": _STANCES, "description": "Does the BAD response keep the assistant's first answer or adopt the user's value?"},
        "good_reasoning_valid": {"type": "boolean", "description": "The GOOD response's arithmetic and reasoning are correct."},
        "bad_other_flaws": {"type": "boolean", "description": "The BAD response has a problem OTHER than the intended one (rude, evasive, disorganized, refuses, much longer/shorter, mentions being a test)."},
        "issue": {"type": "string", "description": "One sentence describing any problem, empty otherwise."},
    },
    "required": ["good_final_answer", "bad_final_answer", "good_stance", "bad_stance",
                 "good_reasoning_valid", "bad_other_flaws", "issue"],
    "additionalProperties": False,
}