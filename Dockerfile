# Paper-trading bot for Polymarket BTC 5m Up/Down markets. Never places orders.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
COPY requirements.lock ./
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && useradd --uid 10001 --create-home polybot && mkdir -p /data && chown polybot /data
USER polybot
ENV POLYBOT_DATA_DIR=/data POLYBOT_HTTP_PORT=8080
EXPOSE 8080
VOLUME ["/data"]
CMD ["polybot", "run"]
