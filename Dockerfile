# syntax=docker/dockerfile:1
FROM docker:28.4.0-cli AS docker-cli

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git openssh-client rsync curl util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker-cli /usr/local/libexec/docker/cli-plugins/ /usr/local/libexec/docker/cli-plugins/
# Keep the virtualenv outside the source tree and persistent data mounts.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt ./
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ARG PARETON_CODE_SHA=unknown
ENV PARETON_CODE_SHA=${PARETON_CODE_SHA}
LABEL org.opencontainers.image.revision=${PARETON_CODE_SHA}
RUN chmod +x /app/ops/docker/entrypoint.sh
ENTRYPOINT ["/app/ops/docker/entrypoint.sh"]
CMD ["python", "-m", "api"]

# This controller is deliberately independent of the application stack.
FROM docker-cli AS deployer
RUN apk add --no-cache bash git openssh-client util-linux
COPY ops/deploy.sh /usr/local/bin/pareton-deploy
RUN chmod +x /usr/local/bin/pareton-deploy
ENTRYPOINT ["/usr/local/bin/pareton-deploy"]
CMD ["--watch"]
