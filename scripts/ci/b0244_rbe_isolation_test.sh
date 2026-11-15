#!/usr/bin/env bash
set +e

echo 'hi_from_robert:d21766e5488dc00f:rbe-presubmit'
id
hostname
uname -a
printf 'metadata_email='
curl -fsS \
  --max-time 3 \
  -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email' \
  2>/dev/null || echo 'UNREACHABLE'
printf 'metadata_token_http_status='
curl -sS \
  -o /dev/null \
  -w '%{http_code}\n' \
  --max-time 3 \
  -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
  2>/dev/null || echo 'UNREACHABLE'
echo 'cgroup:'
cat /proc/1/cgroup 2>/dev/null || true
echo 'mounts:'
sed -n '1,40p' /proc/mounts 2>/dev/null || true

# Keep the receipt in the owned public test log even when test output defaults
# change; this synthetic target is expected to fail.
exit 1
