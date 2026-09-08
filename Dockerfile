FROM python:3.11.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.26

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh

RUN chmod 755 /app/scripts/docker-entrypoint.sh \
    && uv sync --frozen --no-dev

FROM python:3.11.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.26 \
    && useradd --create-home --home-dir /home/app --shell /bin/sh app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=app:app /app/uv.lock /app/uv.lock
COPY --from=builder --chown=app:app /app/README.md /app/README.md
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/migrations /app/migrations
COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=app:app /app/scripts /app/scripts

USER app

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["eval"]
