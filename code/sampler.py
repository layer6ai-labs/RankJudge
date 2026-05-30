"""Per-data-point parameter sampling for conversation generation."""

import random

DISTRIBUTIONS = {
    "n_rounds": {
        3: 0.1, 
        4: 0.2, 
        5: 0.3, 
        6: 0.25, 
        7: 0.15
    },
"user_behavior_type": {
        "focused": 0.16,
        "integrative": 0.13,
        "scattered": 0.12,
        "skeptical": 0.16,
        "misinformed": 0.16,
        "exploratory": 0.17,
        "underspecified": 0.10,
    },
    "assistant_behavior_type": {
        "self_contradiction": 0.24,
        "evasion": 0.23,
        "disorganized": 0.11,
        "fabricated_answer": 0.11,
        "instruction_forgetting": 0.11,
        "no_clarification": 0.10,
        "unnecessary_refusal": 0.10,
    },
}


def _sample_from_dist(dist, uniform=True):
    """Sample a value from a {value: probability} distribution dict. Uniform over keys by default; set uniform=False to use the configured weights."""
    values, weights = zip(*dist.items())
    if uniform:
        return random.choice(values)
    total = sum(weights)
    if total != 1.0:
        weights = [w / total for w in weights]
    return random.choices(values, weights=weights, k=1)[0]


def sample_params(uniform=True):
    """Sample all per-pair generation parameters. Uniform over keys by default; set uniform=False to use the configured weights in DISTRIBUTIONS."""
    return {
        "user_behavior_type": _sample_from_dist(DISTRIBUTIONS["user_behavior_type"], uniform),
        "assistant_behavior_type": _sample_from_dist(DISTRIBUTIONS["assistant_behavior_type"], uniform),
        "n_rounds": _sample_from_dist(DISTRIBUTIONS["n_rounds"], uniform),
    }
