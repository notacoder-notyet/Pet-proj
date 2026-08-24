# Образ для инференса / Inference image
FROM python:3.12-slim

WORKDIR /app

# Не-root пользователь / Non-root user
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

# Зависимости отдельно — лучше кэшируется слой / Install deps first for better layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/predict.py"]
