# Образ для API и бота / Image for API and bot
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Не-root пользователь / Non-root user
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# По умолчанию REST API; бот задаётся в docker-compose / Default REST API; bot is set in compose
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
