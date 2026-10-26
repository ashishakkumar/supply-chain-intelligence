FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY app ./app
COPY db ./db
COPY data ./data
COPY models ./models

RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1 \
    DATA_DIR=/app/data/raw \
    MODEL_PATH=/app/models/delay_model.joblib \
    APP_VERSION=0.1.0

EXPOSE 8000

CMD ["sh", "-c", "uv run db-migrate && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
