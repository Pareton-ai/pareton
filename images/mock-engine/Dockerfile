# Minimal mock OpenAI-compatible engine for bench lifecycle / correctness CI.
# Build from repo root:
#   docker build -f images/mock-engine/Dockerfile -t pareton-mock-engine:local .
FROM python:3.12-slim

WORKDIR /app
COPY bench/__init__.py bench/mock_engine.py /app/bench/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "-m", "bench.mock_engine", "--host", "0.0.0.0", "--port", "8000"]
