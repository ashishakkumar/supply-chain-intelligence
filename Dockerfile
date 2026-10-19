FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY models ./models

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data/raw \
    MODEL_PATH=/app/models/delay_model.joblib \
    APP_VERSION=0.1.0

EXPOSE 8000

CMD ["sh", "-c", "python -m app.pipeline && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
