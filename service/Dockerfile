# Built and pushed with --platform=linux/amd64 explicitly: Apple Silicon
# builds an arm64 image by default, and Cloud Run rejects arm64 outright.
# See infra/setup.sh.
FROM python:3.12-slim

WORKDIR /app

COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY config.yaml ./config.yaml
COPY src ./src
COPY service ./service
# The frozen reference this deployment serves. Baked into the image rather
# than fetched from GCS at startup, since it is small (about 1.4 MB) and
# this keeps the deployed service self-contained: no runtime IAM/GCS
# dependency just to load the one thing every request needs.
COPY artifacts ./artifacts

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn service.main:app --host 0.0.0.0 --port ${PORT}"]
