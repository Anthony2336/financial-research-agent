"""Public immutable prompt registry used by provider calls and run provenance."""

from fra.prompts.analyst import (
    ANALYST_PROMPT,
    REPAIR_PROMPT,
    THESIS_ANALYST_PROMPT,
)
from fra.prompts.planner import (
    ROUTER_PROMPT,
    SKILL_PLANNER_PROMPT,
    THESIS_PLANNER_PROMPT,
)
from fra.prompts.system import (
    RESEARCH_PROMPT_VERSION,
    STRICT_GROUNDING_SYSTEM,
    PromptBundle,
    PromptUsage,
    bind_prompt_usage,
    current_prompt_version,
    record_prompt_version,
)

__all__ = [
    "ANALYST_PROMPT",
    "REPAIR_PROMPT",
    "RESEARCH_PROMPT_VERSION",
    "ROUTER_PROMPT",
    "SKILL_PLANNER_PROMPT",
    "STRICT_GROUNDING_SYSTEM",
    "THESIS_ANALYST_PROMPT",
    "THESIS_PLANNER_PROMPT",
    "PromptBundle",
    "PromptUsage",
    "bind_prompt_usage",
    "current_prompt_version",
    "record_prompt_version",
]
