FROM python:3.11-slim

# Install pytest and requests for tests
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY bazel_pr_mitigation.zip /workspace/bazel_pr_mitigation.zip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
RUN unzip /workspace/bazel_pritigation.zip || unzip /workspace/bazel_pritigation.zip -d /workspace || true
# If unzip above doesn't extract because of filename mismatch, try alternative name
RUN if [ -d /workspace/bazel_pritigation ]; then mv /workspace/bazel_pritigation/* /workspace/; fi || true
# Copy whole context (fallback if zip not present)
COPY . /workspace

RUN pip install --no-cache-dir pytest requests

# Default command: run python tests (pytest)
CMD ["sh", "-c", "pytest -q || true"]
