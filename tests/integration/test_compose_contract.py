import json
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def _resolved_compose_config(*, profile: str | None = None) -> dict[str, object]:
    docker_cli = os.environ.get("DOCKER_CLI", "docker")
    command = [docker_cli, "compose"]
    if profile is not None:
        command.extend(["--profile", profile])
    command.extend(["config", "--format", "json"])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_compose_declares_only_required_p0_services() -> None:
    config = _resolved_compose_config(profile="cli")

    services = config["services"]
    assert set(services) == {"postgres", "redis", "cli"}
    assert services["postgres"]["image"] == "pgvector/pgvector:pg16"
    assert services["redis"]["image"] == "redis:7-alpine"
    assert services["postgres"]["healthcheck"]["test"]
    assert services["redis"]["healthcheck"]["test"]
    assert services["cli"]["profiles"] == ["cli"]
    assert services["cli"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["cli"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert services["cli"]["environment"]["DATABASE_URL"].startswith(
        "postgresql+psycopg://financial_evidence:financial_evidence@postgres:5432/"
    )
    assert services["cli"]["environment"]["REDIS_URL"] == "redis://redis:6379/0"
    assert services["cli"]["environment"]["TOKENIZER_CACHE_DIR"] == (
        "/opt/financial-evidence-agent/models/tokenizer"
    )
    assert services["cli"]["environment"]["CONTEXT_COMPRESSOR_CACHE_DIR"] == (
        "/opt/financial-evidence-agent/models/llmlingua"
    )
    assert services["cli"]["environment"]["RERANKER_CACHE_DIR"] == (
        "/opt/financial-evidence-agent/models/flashrank"
    )
    assert services["cli"]["environment"]["MARKET_CONTEXT_WINDOW_DAYS"] == "3"
    assert services["cli"]["environment"]["MARKET_ABNORMAL_MOVE_THRESHOLD"] == "0.05"
    assert services["cli"]["environment"]["RESEARCH_QUALITY_MAX_SOURCE_AGE_DAYS"] == "365"
    assert services["cli"]["environment"]["RESEARCH_MEMORY_TTL_DAYS"] == "90"
    model_mounts = {
        volume["target"]: volume
        for volume in services["cli"]["volumes"]
        if volume["target"].startswith("/opt/financial-evidence-agent/models/")
    }
    assert set(model_mounts) == {
        "/opt/financial-evidence-agent/models/embedding",
        "/opt/financial-evidence-agent/models/tokenizer",
        "/opt/financial-evidence-agent/models/llmlingua",
        "/opt/financial-evidence-agent/models/flashrank",
    }
    assert all(volume["type"] == "bind" for volume in model_mounts.values())
    assert all(volume["read_only"] is True for volume in model_mounts.values())
    assert "ports" not in services["cli"]


def test_environment_template_documents_p0_integrations() -> None:
    values = dotenv_values(Path(".env.example"))

    assert values["DATABASE_URL"].startswith("postgresql+psycopg://")
    assert values["REDIS_URL"] == "redis://localhost:6379/0"
    assert values["SEC_USER_AGENT"] == "Example Research Operator research-operator@example.com"
    assert {"FAST_MODEL", "ANALYST_MODEL", "OPENAI_API_KEY"} <= values.keys()
    assert {"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"} <= values.keys()
    assert values["EMBEDDING_CACHE_DIR"] == ".model-cache/bge-m3"
    assert values["TOKENIZER_CACHE_DIR"] == ".model-cache/tokenizer"
    assert values["CONTEXT_COMPRESSOR_CACHE_DIR"] == ".model-cache/llmlingua"
    assert values["RERANKER_CACHE_DIR"] == ".model-cache/flashrank"
