"""Random choices made per pair."""

import random


def sample_repair_params(strategy="candidate"):
    """Pick a phrasing template for the user's repair turn (turn 3)."""
    from taxonomy import REPAIR_STRATEGIES
    templates = REPAIR_STRATEGIES[strategy]
    idx = random.randrange(len(templates))
    return {"repair_strategy": strategy, "repair_template_idx": idx,
            "repair_template": templates[idx]}
