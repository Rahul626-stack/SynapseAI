"""Bloom's Taxonomy Engine — cognitive level distribution and prompt engineering."""

from enum import Enum
from typing import Dict, Optional


class BloomLevel(str, Enum):
    REMEMBER   = "Remember"
    UNDERSTAND = "Understand"
    APPLY      = "Apply"
    ANALYZE    = "Analyze"
    EVALUATE   = "Evaluate"
    CREATE     = "Create"


# Verb hints for prompting — used to guide the LLM toward appropriate cognitive tasks
BLOOM_VERBS = {
    BloomLevel.REMEMBER:   "recall, list, define, identify, name",
    BloomLevel.UNDERSTAND: "explain, summarize, paraphrase, classify, describe",
    BloomLevel.APPLY:      "use, solve, demonstrate, compute, implement",
    BloomLevel.ANALYZE:    "compare, differentiate, examine, break down, infer",
    BloomLevel.EVALUATE:   "judge, justify, critique, argue, assess",
    BloomLevel.CREATE:     "design, construct, formulate, propose, devise",
}

# Default distribution for a 10-question quiz
DEFAULT_DISTRIBUTION = {
    BloomLevel.REMEMBER:   2,
    BloomLevel.UNDERSTAND: 3,
    BloomLevel.APPLY:      2,
    BloomLevel.ANALYZE:    2,
    BloomLevel.EVALUATE:   1,
    BloomLevel.CREATE:     0,
}

BLOOM_ORDER = [
    BloomLevel.REMEMBER,
    BloomLevel.UNDERSTAND,
    BloomLevel.APPLY,
    BloomLevel.ANALYZE,
    BloomLevel.EVALUATE,
    BloomLevel.CREATE,
]


def scale_distribution(
    target_total: int,
    base: Optional[Dict[BloomLevel, int]] = None
) -> Dict[str, int]:
    """Proportionally scale the default Bloom's distribution to any target question count.

    Uses proportional rounding with remainder redistribution to ensure
    the total matches exactly.

    Args:
        target_total: The desired number of questions.
        base: Optional custom base distribution. Defaults to DEFAULT_DISTRIBUTION.

    Returns:
        A dict mapping Bloom level name (str) → question count (int).
    """
    base = base or DEFAULT_DISTRIBUTION
    base_total = sum(base.values())

    if base_total == 0:
        return {level.value: 0 for level in BLOOM_ORDER}

    # Proportional scaling
    scaled = {}
    for level in BLOOM_ORDER:
        count = base.get(level, 0)
        scaled[level.value] = int((count / base_total) * target_total)

    # Distribute remainders to match target_total exactly
    current_total = sum(scaled.values())
    remainder = target_total - current_total

    # Sort levels by their original proportion (descending) for remainder distribution
    sorted_levels = sorted(
        [(level.value, base.get(level, 0)) for level in BLOOM_ORDER],
        key=lambda x: x[1],
        reverse=True,
    )

    i = 0
    while remainder > 0:
        level_name = sorted_levels[i % len(sorted_levels)][0]
        scaled[level_name] += 1
        remainder -= 1
        i += 1

    return scaled


def build_bloom_prompt_instructions(
    num_questions: int = 10,
    distribution: Optional[Dict[str, int]] = None,
) -> str:
    """Generate structured prompt block with per-level quotas and verb hints.

    This string is injected directly into the quiz generator task prompt
    to enforce Bloom's Taxonomy distribution.

    Args:
        distribution: Optional pre-computed distribution dict.
        num_questions: Target question count (used if distribution is None).

    Returns:
        A multi-line prompt instruction string for the LLM.
    """
    if distribution is None:
        distribution = scale_distribution(num_questions)

    lines = ["BLOOM'S TAXONOMY DISTRIBUTION REQUIREMENT:"]
    lines.append(
        "You MUST generate exactly the following number of questions "
        "per cognitive level:\n"
    )

    for level in BLOOM_ORDER:
        level_name = level.value
        count = distribution.get(level_name, 0)
        if count > 0:
            verbs = BLOOM_VERBS.get(level, "")
            lines.append(
                f"- {level_name}: {count} question(s). "
                f"(Use verbs like: {verbs})"
            )

    lines.append(
        '\nFor each question in your output, add a field: "bloom_level": "<Level>" '
        "so the level is explicitly tagged."
    )

    return "\n".join(lines)
