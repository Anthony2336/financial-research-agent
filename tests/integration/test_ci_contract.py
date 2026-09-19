"""Release-delivery contract tests for CI policy, live gating, and operator docs."""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import yaml
from dotenv import dotenv_values


def _load_workflow(name: str) -> tuple[dict[str, object], str]:
    path = Path(".github/workflows") / name
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded, text


def _workflow_on(workflow: dict[str, object]) -> object:
    return workflow.get("on", workflow.get(True))


def _load_project_conftest():
    path = Path("tests/conftest.py")
    spec = importlib.util.spec_from_file_location("project_conftest", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _job_run_text(workflow: dict[str, object], job_name: str) -> str:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[job_name]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    return "\n".join(
        str(step.get("run", ""))
        for step in steps
        if isinstance(step, dict)
    )


def _job_steps(workflow: dict[str, object], job_name: str) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[job_name]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    named_steps = {
        step["name"]: step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }
    return named_steps


def _git_subprocess_environment(*, base_sha: str | None = None) -> dict[str, str]:
    """Return a Git-only environment that cannot inherit repository routing."""
    environment = {
        name: value
        for name in ("PATH", "SYSTEMROOT", "WINDIR", "ComSpec")
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    if base_sha is not None:
        environment["BASE_SHA"] = base_sha
    return environment


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        env=_git_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repository: Path, filename: str, content: bytes, message: str) -> str:
    (repository / filename).write_bytes(content)
    _git(repository, "add", filename)
    _git(repository, "commit", "-qm", message)
    return _git(repository, "rev-parse", "HEAD")


def _new_git_repository(path: Path) -> tuple[Path, str]:
    repository = path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "ci-contract@example.test")
    _git(repository, "config", "user.name", "CI Contract")
    base_sha = _commit(repository, "evidence.txt", b"clean base\n", "base")
    return repository, base_sha


def _run_diff_whitespace_step(
    run: str, repository: Path, base_sha: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", run],
        cwd=repository,
        env=_git_subprocess_environment(base_sha=base_sha),
        check=False,
        capture_output=True,
        text=True,
    )


def test_default_ci_runs_only_offline_release_checks() -> None:
    workflow, text = _load_workflow("ci.yml")
    run_text = _job_run_text(workflow, "offline-validation")
    steps = _job_steps(workflow, "offline-validation")
    trigger = _workflow_on(workflow)

    assert isinstance(trigger, dict)
    assert {"push", "pull_request"} <= set(trigger)
    assert "OPENAI_API_KEY" not in text
    assert "ALPACA_API_SECRET_KEY" not in text
    assert "LANGFUSE_SECRET_KEY" not in text
    assert "TAVILY_API_KEY" not in text
    assert "prefetch-model-assets" not in run_text
    assert "uv sync --frozen --all-groups" in run_text
    assert 'uv run pytest -m "not live_sec and not live_provider" -q' in run_text
    assert "uv run eval" in run_text
    assert "uv run eval --suite p2" in run_text
    assert "uv build --offline --no-build-isolation" in run_text
    postgres_acceptance = steps["PostgreSQL Acceptance"]
    assert postgres_acceptance["env"] == {"RUN_POSTGRES_INTEGRATION": "1"}
    assert "tests/integration/test_database_migrations.py" in postgres_acceptance["run"]
    assert "tests/integration/test_postgres_fixture_ingest.py" in postgres_acceptance["run"]
    assert "tests/integration/test_postgres_vector_retrieval.py" in postgres_acceptance["run"]
    assert "docker build -t financial-evidence-agent:ci ." not in run_text
    assert "docker compose config --quiet" in run_text
    assert "docker compose up -d --wait postgres redis" in run_text
    assert run_text.count("docker compose --profile cli build cli") == 1
    assert "docker compose --profile cli run --rm cli db-upgrade" in run_text
    assert (
        "docker compose --profile cli run --rm cli ingest --fixture "
        "/app/src/fra/resources/nvda_10q.html"
    ) in run_text
    assert "docker compose --profile cli run --rm cli research NVDA --thesis" in run_text
    assert "docker compose --profile cli run --rm cli eval --suite p2" in run_text
    assert "documented target: <60s" in run_text
    assert "test \"$elapsed_seconds\" -lt 60" in run_text
    assert "retrieval_cache_key=" in run_text
    assert "test -n \"$retrieval_cache_key\"" in run_text
    assert "cache_hits_before=" in run_text
    assert "cache_hits_after=" in run_text
    assert "test \"$cache_hits_after\" -gt \"$cache_hits_before\"" in run_text
    research = "docker compose --profile cli run --rm cli research NVDA --thesis"
    first_research = run_text.index(research)
    cache_key_check = run_text.index("retrieval_cache_key=", first_research)
    hits_before = run_text.index("cache_hits_before=", cache_key_check)
    timed_second = run_text.index("start_seconds=", hits_before)
    second_research = run_text.index(research, timed_second)
    hits_after = run_text.index("cache_hits_after=", second_research)
    assert first_research < cache_key_check < hits_before < timed_second
    assert timed_second < second_research < hits_after


def test_ci_whitespace_gate_compares_a_validated_commit_range() -> None:
    """Reject a workflow change that checks only the runner working tree."""
    workflow, _ = _load_workflow("ci.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["offline-validation"]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    named_steps = _job_steps(workflow, "offline-validation")

    checkout = next(
        step
        for step in steps
        if isinstance(step, dict) and step.get("uses") == "actions/checkout@v4"
    )
    assert checkout.get("with") == {"fetch-depth": 0}

    whitespace = named_steps["Diff Whitespace"]
    assert whitespace.get("env") == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha || github.event.before }}"
    }
    run = whitespace.get("run")
    assert isinstance(run, str)
    assert 'git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null' in run
    assert "git rev-list --parents HEAD" in run
    assert "git hash-object -t tree /dev/null" in run
    assert "git rev-parse --verify HEAD^" not in run
    assert "exit 1" in run
    diff_checks = [
        line.strip()
        for line in run.splitlines()
        if line.strip().startswith("git diff --check")
    ]
    assert diff_checks == [
        'git diff --check "$BASE_SHA" HEAD',
        'git diff --check "$EMPTY_TREE" "$commit" || exit 1',
        'git diff --check "$parent" "$commit" || exit 1',
    ]

    step_names = [step.get("name") for step in steps if isinstance(step, dict)]
    assert step_names.index("Diff Whitespace") < step_names.index("Sync Dependencies")


def test_ci_whitespace_step_executes_the_event_range_and_fails_closed() -> None:
    """Reject an unreachable diff command or one that ignores the event base SHA."""
    workflow, _ = _load_workflow("ci.yml")
    run = _job_steps(workflow, "offline-validation")["Diff Whitespace"]["run"]
    assert isinstance(run, str)

    with TemporaryDirectory() as directory:
        root = Path(directory)

        clean_repository, clean_base = _new_git_repository(root / "clean")
        _commit(clean_repository, "evidence.txt", b"clean head\n", "clean head")
        clean = _run_diff_whitespace_step(run, clean_repository, clean_base)
        assert clean.returncode == 0, clean.stderr

        event_repository, event_base = _new_git_repository(root / "event-base")
        _commit(event_repository, "evidence.txt", b"trailing whitespace \n", "bad range")
        _commit(event_repository, "clean.txt", b"clean head\n", "clean head")
        event_base_failure = _run_diff_whitespace_step(run, event_repository, event_base)
        assert event_base_failure.returncode != 0
        assert "trailing whitespace" in event_base_failure.stdout

        fallback_repository, _ = _new_git_repository(root / "fallback")
        _commit(fallback_repository, "evidence.txt", b"trailing whitespace \n", "bad middle")
        _commit(fallback_repository, "evidence.txt", b"clean head\n", "clean head")
        fallback = _run_diff_whitespace_step(run, fallback_repository, "0" * 40)
        assert fallback.returncode != 0
        assert "trailing whitespace" in fallback.stdout

        initial_repository, _ = _new_git_repository(root / "initial")
        _git(initial_repository, "checkout", "--orphan", "initial-only")
        _git(initial_repository, "rm", "-q", "-rf", ".")
        _commit(initial_repository, "evidence.txt", b"initial commit\n", "initial")
        initial = _run_diff_whitespace_step(run, initial_repository, "0" * 40)
        assert initial.returncode == 0, initial.stderr


def test_ci_whitespace_step_ignores_poisoned_git_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extracted step must operate only on its temporary repository."""
    workflow, _ = _load_workflow("ci.yml")
    run = _job_steps(workflow, "offline-validation")["Diff Whitespace"]["run"]
    assert isinstance(run, str)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        repository, base_sha = _new_git_repository(root / "target")
        _commit(repository, "evidence.txt", b"clean head\n", "clean head")
        poisoned_repository, _ = _new_git_repository(root / "poisoned")
        _commit(
            poisoned_repository,
            "evidence.txt",
            b"poisoned trailing whitespace \n",
            "poisoned head",
        )
        poisoned_config = root / "poisoned.gitconfig"
        poisoned_config.write_text(
            f"[core]\n\tworktree = {poisoned_repository}\n", encoding="utf-8"
        )

        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(poisoned_config))
        monkeypatch.setenv("GIT_DIR", str(poisoned_repository / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(poisoned_repository))
        monkeypatch.setenv("GIT_INDEX_FILE", str(root / "poisoned.index"))
        monkeypatch.setenv(
            "GIT_OBJECT_DIRECTORY", str(poisoned_repository / ".git" / "objects")
        )
        monkeypatch.setenv(
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            str(poisoned_repository / ".git" / "objects"),
        )

        result = _run_diff_whitespace_step(run, repository, base_sha)

    assert result.returncode == 0, result.stderr


def test_live_smoke_workflow_is_manual_and_secret_scoped() -> None:
    workflow, text = _load_workflow("live-smoke.yml")
    trigger = _workflow_on(workflow)
    run_text = _job_run_text(workflow, "live-smoke")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["live-smoke"]
    assert isinstance(job, dict)

    assert isinstance(trigger, dict)
    assert set(trigger) == {"workflow_dispatch"}
    assert job["environment"] == "live-smoke"
    assert job.get("env") == {"UV_CACHE_DIR": ".uv-cache"}
    dispatch = trigger["workflow_dispatch"]
    assert isinstance(dispatch, dict)
    inputs = dispatch["inputs"]
    assert isinstance(inputs, dict)
    assert {
        "run_p1_smoke",
        "run_alpaca_smoke",
        "run_langfuse_smoke",
        "run_sec_smoke",
        "run_tavily_smoke",
    } <= set(inputs)
    steps = _job_steps(workflow, "live-smoke")
    assert set(steps) >= {
        "Sync Dependencies",
        "Sync Web-Search Extra",
        "Prefetch Local Model Assets",
        "P1 Application Smoke",
        "Alpaca IEX Smoke",
        "Langfuse Smoke",
        "SEC Smoke",
        "Tavily Smoke",
    }
    assert steps["Sync Dependencies"].get("env") is None
    assert steps["Sync Web-Search Extra"].get("env") is None
    assert steps["Prefetch Local Model Assets"]["env"] == {
        "FAST_MODEL": "${{ vars.FAST_MODEL }}",
        "ANALYST_MODEL": "${{ vars.ANALYST_MODEL }}",
        "EMBEDDING_CACHE_DIR": ".model-cache/bge-m3",
        "TOKENIZER_CACHE_DIR": ".model-cache/tokenizer",
        "CONTEXT_COMPRESSOR_CACHE_DIR": ".model-cache/llmlingua",
        "RERANKER_CACHE_DIR": ".model-cache/flashrank",
    }
    assert steps["P1 Application Smoke"]["env"] == {
        "OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}",
        "FAST_MODEL": "${{ vars.FAST_MODEL }}",
        "ANALYST_MODEL": "${{ vars.ANALYST_MODEL }}",
        "EMBEDDING_CACHE_DIR": ".model-cache/bge-m3",
        "TOKENIZER_CACHE_DIR": ".model-cache/tokenizer",
        "CONTEXT_COMPRESSOR_CACHE_DIR": ".model-cache/llmlingua",
        "RERANKER_CACHE_DIR": ".model-cache/flashrank",
    }
    step_order = list(steps)
    assert step_order.index("Prefetch Local Model Assets") < step_order.index(
        "P1 Application Smoke"
    )
    assert "secrets." not in repr(steps["Prefetch Local Model Assets"]["env"])
    assert steps["Alpaca IEX Smoke"]["env"] == {
        "ALPACA_API_KEY_ID": "${{ secrets.ALPACA_API_KEY_ID }}",
        "ALPACA_API_SECRET_KEY": "${{ secrets.ALPACA_API_SECRET_KEY }}",
        "ALPACA_TRADING_ENVIRONMENT": "${{ vars.ALPACA_TRADING_ENVIRONMENT }}",
    }
    assert steps["Langfuse Smoke"]["env"] == {
        "LANGFUSE_PUBLIC_KEY": "${{ secrets.LANGFUSE_PUBLIC_KEY }}",
        "LANGFUSE_SECRET_KEY": "${{ secrets.LANGFUSE_SECRET_KEY }}",
        "LANGFUSE_HOST": "${{ vars.LANGFUSE_HOST }}",
    }
    assert steps["SEC Smoke"]["env"] == {
        "SEC_USER_AGENT": "${{ vars.SEC_USER_AGENT }}"
    }
    assert steps["Tavily Smoke"]["env"] == {
        "TAVILY_API_KEY": "${{ secrets.TAVILY_API_KEY }}"
    }
    assert "live_openai_skill_models_smoke" not in run_text
    assert "live_p1_application_smoke" in run_text
    assert "tests/integration/test_live_alpaca_opt_in.py" in run_text
    assert "tests/integration/test_live_langfuse_opt_in.py" in run_text
    assert "tests/integration/test_live_sec_opt_in.py" in run_text
    assert "-m live_provider" in run_text
    assert "printenv" not in text
    assert "env |" not in text


def test_live_markers_are_opt_in_and_network_stays_blocked_by_default() -> None:
    project_conftest = _load_project_conftest()
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "addopts = \"-m 'not live_sec and not live_provider'\"" in pyproject

    configured_markers: list[tuple[str, str]] = []

    class ConfigRecorder:
        def addinivalue_line(self, section: str, value: str) -> None:
            configured_markers.append((section, value))

    project_conftest.pytest_configure(ConfigRecorder())

    assert (
        "markers",
        "live_sec: allows an explicitly requested live SEC test",
    ) in configured_markers
    assert (
        "markers",
        "live_provider: allows explicitly selected live provider connectivity tests",
    ) in configured_markers

    with pytest.MonkeyPatch.context() as isolated:
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect
        unmarked = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: None))
        guard = project_conftest.block_external_network.__wrapped__(unmarked, isolated)
        next(guard)
        with pytest.raises(
            RuntimeError,
            match="external network disabled during tests: example.com",
        ):
            socket.getaddrinfo("example.com", 443)
        assert socket.socket.connect is not original_connect
        next(guard, None)

    with pytest.MonkeyPatch.context() as isolated:
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect
        live_provider = SimpleNamespace(
            node=SimpleNamespace(
                get_closest_marker=lambda name: object() if name == "live_provider" else None
            )
        )
        passthrough = project_conftest.block_external_network.__wrapped__(
            live_provider, isolated
        )
        next(passthrough)
        assert socket.getaddrinfo is original_getaddrinfo
        assert socket.socket.connect is original_connect
        next(passthrough, None)


def test_environment_template_documents_current_release_capabilities() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    values = dotenv_values(Path(".env.example"))

    assert values["LANGFUSE_HOST"] == "https://cloud.langfuse.com"
    assert values["MARKET_DATA_PROVIDER"] == "alpaca"
    assert values["ALPACA_DATA_FEED"] == "iex"
    assert values["MARKET_CONTEXT_WINDOW_DAYS"] == "3"
    assert values["RESEARCH_QUALITY_MAX_SOURCE_AGE_DAYS"] == "365"
    assert "OPENAI_API_KEY=" in text
    assert "TAVILY_API_KEY=" in text
    assert "LANGFUSE_PUBLIC_KEY=" in text
    assert "LANGFUSE_SECRET_KEY=" in text
    assert "ALPACA_API_KEY_ID=" in text
    assert "ALPACA_API_SECRET_KEY=" in text
    assert "P0 / P1 / P2" in text
    assert "LANGFUSE_HOST=https://cloud.langfuse.com" in text


def test_operator_guide_includes_non_destructive_legacy_compose_recovery_sequence() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    fresh_project = "financial-evidence-agent-legacy-recovery"
    preserve = "docker compose -p financial-evidence-agent stop postgres redis"
    fresh_up = f"docker compose -p {fresh_project} up -d postgres redis"
    build = f"docker compose -p {fresh_project} --profile cli build cli"
    migrate = f"docker compose -p {fresh_project} --profile cli run --rm cli db-upgrade"
    ingest = (
        f"docker compose -p {fresh_project} --profile cli run --rm cli ingest "
        "--fixture /app/src/fra/resources/nvda_10q.html "
        "--ticker NVDA --form 10-Q"
    )
    research = (
        f"docker compose -p {fresh_project} --profile cli run --rm cli research NVDA "
        '--thesis "Do cited disclosures support sustained data center demand?"'
    )
    evaluate = f"docker compose -p {fresh_project} --profile cli run --rm cli eval --suite p2"
    cleanup = f"docker compose -p {fresh_project} down"
    restart = "docker compose -p financial-evidence-agent up -d postgres redis"
    destructive = "docker compose -p financial-evidence-agent-release-gate-20260831 down -v"

    for line in (preserve, fresh_up, build, migrate, ingest, research, evaluate, restart, cleanup):
        assert line in text
    assert text.index(preserve) < text.index(fresh_up) < text.index(build) < text.index(migrate)
    assert text.index(migrate) < text.index(ingest) < text.index(research) < text.index(evaluate)
    assert (
        text.index(evaluate)
        < text.index(cleanup)
        < text.index(restart)
        < text.index(destructive)
    )


def test_readme_distinguishes_in_process_cli_from_standalone_stdio() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "in-process FastMCP" in text
    assert "standalone stdio" in text
    assert "starts an MCP subprocess for every CLI command" not in text


def test_readme_links_to_engineering_highlights() -> None:
    highlights_path = Path("PROJECT_HIGHLIGHTS.md")
    assert highlights_path.is_file()

    text = Path("README.md").read_text(encoding="utf-8")
    readme_targets = {
        Path(target)
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    }
    assert highlights_path in readme_targets
    assert (Path("README.md").parent / highlights_path).is_file()
