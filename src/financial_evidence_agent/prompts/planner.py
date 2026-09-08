"""Versioned router and planner task prompts."""

from financial_evidence_agent.prompts.system import prompt_bundle

ROUTER_PROMPT = prompt_bundle(
    "Classify the request into exactly one allowed intent and give a concise reason. "
    "Use only the closed intent enum in task_data. You must not analyze the security, "
    "answer the request, select data sources, or call tools."
)

THESIS_PLANNER_PROMPT = prompt_bundle(
    "Plan at most three structured research questions for this thesis. Each question "
    "must include separate filing queries for supporting and challenging evidence. "
    "Do not select tools, URLs, tickers, or a source policy."
)

SKILL_PLANNER_PROMPT = prompt_bundle(
    "Plan structured research questions for the supplied ticker and request. The recipe "
    "identity, required facets, source policy, and question budget are fixed; do not "
    "select tools, URLs, or a different recipe. Do not predict stock or share prices, "
    "price targets, or future security values; historical prices and disclosed operating "
    "guidance may only be described neutrally."
)
