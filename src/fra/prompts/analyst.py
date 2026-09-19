"""Versioned analyst and future bounded-repair task prompts."""

from fra.prompts.system import prompt_bundle

THESIS_ANALYST_PROMPT = prompt_bundle(
    "Produce a structured thesis audit using only the supplied evidence. Keep support, "
    "counter-evidence, inference, and open questions distinct. Cite filing and web IDs "
    "only from their matching allowed lists. Never invent a source or call a tool."
)

ANALYST_PROMPT = prompt_bundle(
    "Produce the structured research memo using only the supplied evidence. Keep the exact "
    "frozen recipe name and version, cover only its declared facets, and cite only IDs from "
    "the matching allowed source-ID lists. Never invent a source. Do not predict stock or "
    "share prices, price targets, or future security values; historical prices and disclosed "
    "operating guidance may only be described neutrally."
)

REPAIR_PROMPT = prompt_bundle(
    "Repair only the supplied structured draft against the supplied citation failures. "
    "Do not add claims, sources, tools, or general knowledge. Remove unsupported content "
    "when it cannot be repaired from the supplied evidence and return only the required schema."
)
