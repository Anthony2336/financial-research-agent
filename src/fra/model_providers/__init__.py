"""Optional provider adapters for structured research models."""

from fra.model_providers.openai import (
    OpenAISkillAnalystModel,
    OpenAISkillPlannerModel,
    ResearchQuestionPlan,
)

__all__ = [
    "OpenAISkillAnalystModel",
    "OpenAISkillPlannerModel",
    "ResearchQuestionPlan",
]
