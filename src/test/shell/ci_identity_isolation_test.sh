#!/usr/bin/env bash
set -euo pipefail

# This test is only meaningful on Bazel's Linux presubmit workers.
if [[ "$(uname -s)" != "Linux" || "${TEST_INSTALL_BASE:-}" != "/var/lib/buildkite-agent/bazeltest/install_base" ]]; then
  echo "CI identity check skipped outside Bazel Linux presubmit"
  exit 0
fi

metadata_root="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default"
expected_email="buildkite@bazel-untrusted.iam.gserviceaccount.com"

if email=$(curl --noproxy '*' --fail --silent --show-error --connect-timeout 2 --max-time 5 \
    -H 'Metadata-Flavor: Google' "${metadata_root}/email"); then
  if [[ "$email" == "$expected_email" ]]; then
    email_matches=true
  else
    email_matches=false
  fi
else
  email_matches=false
fi

identity_status=not_requested
if [[ "$email_matches" == true ]]; then
  # Discard the response; the audience is inert and no credential is logged.
  identity_url="${metadata_root}/identity?audience=https%3A%2F%2Fexample.invalid%2Fci-check-560628549"
  if identity_status=$(curl --noproxy '*' --silent --show-error --connect-timeout 2 --max-time 5 \
      --output /dev/null --write-out '%{http_code}' \
      -H 'Metadata-Flavor: Google' "$identity_url"); then
    :
  else
    identity_status=request_failed
  fi
fi

echo "CI identity check: expected_service_account=${email_matches} identity_endpoint_http=${identity_status}"
# Deliberately fail so the diagnostic line appears in the public test log.
exit 1
