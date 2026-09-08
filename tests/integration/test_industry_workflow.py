"""Offline integration coverage for single-ticker industry execution."""

from __future__ import annotations

from financial_evidence_agent.research_packages.industry import run_industry_subrun
from financial_evidence_agent.skills.models import SkillName
from financial_evidence_agent.skills.recipes import INDUSTRY_RESEARCH

from .test_p1_workflow import FakeCollector, P1Recorder, _dependencies


def test_industry_subrun_executes_only_the_single_registered_recipe() -> None:
    recorder = P1Recorder()
    dependencies, mcp_client = _dependencies(recorder)

    package = run_industry_subrun("NVDA", "Describe the accelerator industry", dependencies)

    assert recorder.planner_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert recorder.collector_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert recorder.analyst_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert [start[1] for start in recorder.repository_starts] == ["industry_research"]
    assert all(start[2] == "1.0.0" for start in recorder.repository_starts)
    assert mcp_client.calls == []
    assert [recipe.name for recipe in package.provenance.recipes] == [
        SkillName.INDUSTRY_RESEARCH
    ]
    assert package.coverage == "complete"
    assert {claim.facet for claim in package.claims} == set(INDUSTRY_RESEARCH.required_facets)


def test_industry_subrun_reports_insufficient_evidence_without_running_other_flows() -> None:
    recorder = P1Recorder()
    collector = FakeCollector(recorder, incomplete_recipe=SkillName.INDUSTRY_RESEARCH)
    dependencies, _ = _dependencies(recorder, collector=collector)

    package = run_industry_subrun("NVDA", "Describe the accelerator industry", dependencies)

    assert recorder.planner_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert recorder.collector_recipes == [SkillName.INDUSTRY_RESEARCH]
    assert recorder.analyst_recipes == []
    assert package.coverage == "insufficient"
    assert package.claims == []
    assert [recipe.name for recipe in package.provenance.recipes] == [
        SkillName.INDUSTRY_RESEARCH
    ]
    assert "Insufficient evidence for industry research." in package.information_gaps
