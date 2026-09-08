from pathlib import Path


def test_dockerfile_installs_locked_project_as_non_root() -> None:
    dockerfile = Path("Dockerfile").read_text()
    assert "FROM python:3.11.13-slim AS builder" in dockerfile
    assert "FROM python:3.11.13-slim AS runtime" in dockerfile
    builder_section, runtime_section = dockerfile.split(
        "FROM python:3.11.13-slim AS runtime",
        maxsplit=1,
    )
    assert dockerfile.count("FROM python:3.11.13-slim") == 2
    assert dockerfile.count("pip install --no-cache-dir uv==0.11.26") == 2
    assert "uv sync --frozen --no-dev --no-install-project" in builder_section
    assert "uv sync --frozen --no-dev" in builder_section
    assert "COPY pyproject.toml uv.lock ./" in builder_section
    assert "COPY README.md ./" in builder_section
    assert "COPY src ./src" in builder_section
    assert "COPY migrations ./migrations" in builder_section
    assert "COPY alembic.ini ./alembic.ini" in builder_section
    assert "COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh" in builder_section
    assert "COPY --from=builder --chown=app:app /app/.venv /app/.venv" in runtime_section
    assert (
        "COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml"
        in runtime_section
    )
    assert "COPY --from=builder --chown=app:app /app/uv.lock /app/uv.lock" in runtime_section
    assert "COPY --from=builder --chown=app:app /app/README.md /app/README.md" in runtime_section
    assert "COPY --from=builder --chown=app:app /app/src /app/src" in runtime_section
    assert (
        "COPY --from=builder --chown=app:app /app/migrations /app/migrations"
        in runtime_section
    )
    assert (
        "COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini"
        in runtime_section
    )
    assert "COPY --from=builder --chown=app:app /app/scripts /app/scripts" in runtime_section
    assert "uv sync" not in runtime_section
    assert "USER app" in runtime_section
    assert 'ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]' in runtime_section
    assert 'CMD ["eval"]' in runtime_section
    assert "EXPOSE" not in dockerfile


def test_dockerignore_excludes_local_state_and_secrets() -> None:
    entries = {
        line.strip()
        for line in Path(".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {
        ".env",
        ".venv",
        ".uv-cache",
        ".pytest_cache",
        ".ruff_cache",
        ".git",
        ".worktrees",
        ".superpowers",
        "dist",
    } <= entries


def test_entrypoint_dispatches_db_upgrade_without_destructive_steps() -> None:
    entrypoint = Path("scripts/docker-entrypoint.sh")
    text = entrypoint.read_text()

    assert entrypoint.stat().st_mode & 0o111
    assert text.startswith("#!/bin/sh\n")
    assert "set -eu" in text
    assert 'if [ "${1:-}" = "db-upgrade" ]; then' in text
    assert "exec uv run --no-sync alembic upgrade head" in text
    assert 'exec uv run --no-sync "$@"' in text
    assert "drop" not in text.lower()
    assert "down -v" not in text
