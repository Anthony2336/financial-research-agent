"""Unit contracts for the thesis-aware deterministic demo adapter."""

from fra.context import MemoryHint
from fra.graph.demo_models import DeterministicDemoFastModel


def test_demo_planner_builds_queries_from_the_requested_thesis() -> None:
    """A fixed canned query would retrieve the same evidence for every question."""
    model = DeterministicDemoFastModel()

    lunar_question = model.plan(
        "NVDA",
        "Does the filing disclose lunar mining revenue on Mars?",
    )[0]
    data_center_question = model.plan(
        "NVDA",
        "Does data center demand support revenue growth?",
    )[0]

    assert "lunar mining revenue" in lunar_question.support_query.casefold()
    assert "data center demand" in data_center_question.support_query.casefold()
    assert lunar_question.support_query != data_center_question.support_query
    assert lunar_question.challenge_query != data_center_question.challenge_query


def test_demo_planner_accepts_planning_only_memory_hints_without_changing_output() -> None:
    """A later session turn must satisfy the FastModel protocol without hint leakage."""
    model = DeterministicDemoFastModel()
    thesis = "Does data center demand support revenue growth?"

    without_hints = model.plan("NVDA", thesis)
    with_hints = model.plan(
        "NVDA",
        thesis,
        memory_hints=(
            MemoryHint(
                text="Prior guarded turn; planning continuity only.",
                identity="session",
                pointer_id="run-prior",
            ),
        ),
    )

    assert with_hints == without_hints
    assert "run-prior" not in repr(with_hints)
