#!/usr/bin/env python3
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepares .gemini/context.json and .gemini/prompt.md for Bazel cherry-pick PR reviews."""

import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile

from review_common import (
    AUTOMATED_DECISION_PREFIX,
    GEMINI_REVIEW_MARKER,
    GUARDRAIL_BANNER_PREFIX,
    LOCKFILE_PATHS,
    PR_NUMBER_RE,
    REPO_RE,
    SHA_RE,
    SHORT_SHA_RE,
    TRUSTED_AUTHOR_ASSOCIATIONS,
    append_step_summary,
    effective_review_state,
    is_safe_repo_rel_path,
    sanitize_untrusted_multiline,
    sanitize_untrusted_str,
)

EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
RELEASE_BRANCH_RE = re.compile(r"^release-([A-Za-z0-9._-]+)$")
MAX_AUTO_CLEAN_FILES = 500
MAX_LOCKFILE_DIFF_LINES = 2000

CHERRY_PICK_TRAILER_RE = re.compile(r"\(cherry picked from commit ([0-9a-fA-F]{7,40})\)")
BAZEL_IO_COMMIT_TRAILER_RE = re.compile(
    r"(?:^|\n)Commit\s+https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/([0-9a-fA-F]{7,40})\b"
)
GITHUB_COMMIT_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/([0-9a-fA-F]{7,40})"
)
PIPER_REV_ID_RE = re.compile(r"PiperOrigin-RevId:\s*(\d+)")
CHANGE_ID_RE = re.compile(r"Change-Id:\s*(I[0-9a-fA-F]{8,64})")
COPYBARA_PR_RE = re.compile(
    r"COPYBARA_INTEGRATE_REVIEW=https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/(\d+)"
)
FORKED_FROM_RE = re.compile(r"Forked from #(\d+)")
BACKPORT_PR_RE = re.compile(r"\b(?:backport|cherry[- ]?pick)(?:s|ed|ing)?(?:\s+of)?\s+#(\d+)\b", re.IGNORECASE)
BARE_FULL_SHA_RE = re.compile(r"\b([0-9a-fA-F]{40})\b")
BACKTICK_SHORT_SHA_RE = re.compile(r"`([0-9a-fA-F]{10,39})`")
TITLE_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
RELEASE_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
FLIP_INCOMPATIBLE_RE = re.compile(r"\bflip(?:s|ped|ping)?\b[^\n]{0,80}--incompatible_[A-Za-z0-9_]+", re.IGNORECASE)

GENERIC_COMMIT_SUBJECTS = frozenset(
    {
        "update lockfile(s)",
        "update lockfiles",
        "update lockfile",
        "update module.bazel.lock",
        "regenerate lockfile",
        "regenerate module.bazel.lock",
        "fix presubmit",
    }
)


def is_valid_release_branch(base_ref):
    """Returns the release version string if base_ref is a safe release-* branch name, else None."""
    if not isinstance(base_ref, str):
        return None
    m = RELEASE_BRANCH_RE.fullmatch(base_ref)
    if not m:
        return None
    ver = m.group(1)
    if ".." in ver or "@{" in ver or ver.startswith(".") or ver.endswith(".") or ver.endswith(".lock"):
        return None
    return ver


def is_patch_release_version(release_version):
    """Returns True if release_version is a maintenance patch release (X.Y.Z with Z > 0)."""
    if not isinstance(release_version, str):
        return False
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", release_version)
    return bool(m and int(m.group(3)) > 0)


def gh_api(endpoint, *, paginate=False, optional=False):
    """Calls `gh api` (streaming NDJSON via `--paginate --jq .[]` when `paginate=True`)."""
    cmd = ["gh", "api", *(["--paginate", "--jq", ".[]"] if paginate else []), endpoint]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if res.returncode != 0:
        if optional:
            return [] if paginate else {}
        raise RuntimeError(f"gh api {endpoint} failed (exit {res.returncode}): {res.stderr.strip()}")
    if paginate:
        return [json.loads(line) for line in res.stdout.splitlines() if line.strip()]
    return json.loads(res.stdout)


def run_git(args, cwd=None, check=False):
    cmd = ["git", "-c", "core.quotePath=false", *args]
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=check
    )


def run_git_bytes(args, cwd=None, check=False):
    return subprocess.run(["git", "-c", "core.quotePath=false", *args], cwd=cwd, capture_output=True, check=check)


def git_log_grep_shas(rev_range, grep_pattern, *, fixed=False, ignore_case=False, limit=5):
    """Returns commit SHAs matching `grep_pattern` in `rev_range`."""
    flags = ["--fixed-strings"] if fixed else ["-E"]
    if ignore_case:
        flags.append("-i")
    res = run_git(["log", rev_range, *flags, f"--grep={grep_pattern}", "--format=%H", "-n", str(limit)])
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if SHA_RE.fullmatch(ln.strip())]


def extract_git_files_to_dir(tree_sha, paths, dest_dir):
    """Extracts regular non-binary files at `paths` from `tree_sha` into `dest_dir` via `git archive`."""
    safe_paths = sorted({p for p in paths if is_safe_repo_rel_path(p)})
    if not safe_paths or not (isinstance(tree_sha, str) and SHA_RE.fullmatch(tree_sha)) or dest_dir.is_symlink():
        return {}
    res = run_git_bytes(["--literal-pathspecs", "archive", "--format=tar", tree_sha, "--", *safe_paths])
    if res.returncode != 0 or not res.stdout:
        return {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    with tarfile.open(fileobj=io.BytesIO(res.stdout), mode="r:") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not is_safe_repo_rel_path(member.name):
                continue
            fobj = tf.extractfile(member)
            if fobj is None or b"\x00" in fobj.read():
                continue
            try:
                tf.extract(member, path=dest_dir, set_attrs=False, filter="data")
                staged[member.name] = str((dest_dir / member.name).resolve())
            except (tarfile.TarError, OSError):
                continue
    return staged


def iter_diff_tree_records(from_sha, to_sha, *, detect_renames=False):
    """Parses `git diff-tree -r -z` records between two tree-ish SHAs.

    Returns a list of `(path, prev_path, old_mode, new_mode, old_sha, new_sha, status_letter)` tuples,
    or None if `diff-tree` fails.
    """
    if not (isinstance(from_sha, str) and SHA_RE.fullmatch(from_sha)):
        return None
    if not (isinstance(to_sha, str) and SHA_RE.fullmatch(to_sha)):
        return None
    flag = "-M" if detect_renames else "--no-renames"
    res = run_git(["diff-tree", "-r", flag, "-z", from_sha, to_sha])
    if res.returncode != 0:
        return None
    tokens = res.stdout.split("\0")
    records = []
    idx = 0
    while idx + 1 < len(tokens):
        meta = tokens[idx]
        if not meta.startswith(":"):
            idx += 1
            continue
        parts = meta[1:].split()
        if len(parts) < 5:
            idx += 2
            continue
        st_letter = parts[4][0]
        if st_letter in ("R", "C") and idx + 2 < len(tokens):
            prev_path, path = tokens[idx + 1], tokens[idx + 2]
            idx += 3
        else:
            prev_path, path = "", tokens[idx + 1]
            idx += 2
        if path:
            records.append((path, prev_path, parts[0], parts[1], parts[2], parts[3], st_letter))
    return records


def get_diff_tree_entries(from_sha, to_sha):
    """Returns `{path: (old_mode, new_mode, old_sha, new_sha, status_letter)}` with `--no-renames`."""
    records = iter_diff_tree_records(from_sha, to_sha, detect_renames=False)
    if records is None:
        return None
    return {path: (om, nm, os_, ns, st) for path, _, om, nm, os_, ns, st in records}


def get_numstat_map(base_sha, head_sha):
    """Returns `{path: (additions, deletions)}` from `git diff --numstat -M -z`."""
    res = run_git(["diff", "--numstat", "-M", "-z", base_sha, head_sha])
    if res.returncode != 0 or not res.stdout:
        return {}
    tokens = res.stdout.split("\0")
    stats = {}
    idx = 0
    while idx < len(tokens) and tokens[idx]:
        parts = tokens[idx].split("\t", 2)
        if len(parts) != 3:
            idx += 1
            continue
        adds = int(parts[0]) if parts[0].isdigit() else 0
        dels = int(parts[1]) if parts[1].isdigit() else 0
        if parts[2] == "" and idx + 2 < len(tokens):
            path = tokens[idx + 2]
            idx += 3
        else:
            path = parts[2]
            idx += 1
        if path:
            stats[path] = (adds, dels)
    return stats


def detect_mainline_ref():
    """Finds the local git ref representing master/mainline (never falling back to HEAD)."""
    for candidate in ("origin/master", "master", "origin/main", "main"):
        if run_git(["rev-parse", "--verify", f"{candidate}^{{commit}}"]).returncode == 0:
            return candidate
    raise RuntimeError("Could not resolve mainline ref (origin/master or master) in local git repository")


def resolve_commit_sha(sha_prefix):
    if not isinstance(sha_prefix, str) or not SHORT_SHA_RE.fullmatch(sha_prefix.strip()):
        return None
    res = run_git(["rev-parse", "--verify", f"{sha_prefix.strip()}^{{commit}}"])
    full = res.stdout.strip() if res.returncode == 0 else ""
    return full if SHA_RE.fullmatch(full) else None


def is_ancestor(ancestor_sha, descendant_ref):
    return run_git(["merge-base", "--is-ancestor", ancestor_sha, descendant_ref]).returncode == 0


def get_commit_parents(commit_sha):
    res = run_git(["rev-list", "--parents", "-n", "1", commit_sha])
    parts = res.stdout.strip().split() if res.returncode == 0 else []
    return parts[1:] if len(parts) > 1 else []


def list_git_commits(base_sha, head_sha):
    """Returns `[{"sha": ..., "author": ..., "message": ...}]` for commits in `base_sha..head_sha`."""
    res = run_git(["log", "--reverse", "--format=%H%x00%an%x00%B%x1e", f"{base_sha}..{head_sha}"])
    if res.returncode != 0 or not res.stdout:
        return []
    commits = []
    for rec in res.stdout.split("\x1e"):
        parts = rec.lstrip("\n").split("\0", 2)
        if len(parts) == 3 and SHA_RE.fullmatch(parts[0].strip()):
            commits.append({"sha": parts[0].strip(), "author": parts[1].strip(), "message": parts[2].strip()})
    return commits


def _commit_message(commit_obj):
    return str(commit_obj.get("message") or (commit_obj.get("commit") or {}).get("message") or "")


def find_revert_on_master(commit_sha, mainline_ref, excluded_shas=None, _visited=None, _depth=0):
    """Returns the SHA of an active revert or rollback of `commit_sha` on `mainline_ref`, or None."""
    if _depth > 6 or not (isinstance(commit_sha, str) and SHA_RE.fullmatch(commit_sha)):
        return None
    visited = set(_visited) if _visited else set()
    if commit_sha in visited:
        return None
    visited.add(commit_sha)

    short_prefix = commit_sha[:12]
    grep_pat = rf"^((Automated )?rollback of commit {short_prefix}[0-9a-f]*|This reverts commit {short_prefix}[0-9a-f]*)"
    excluded = excluded_shas or set()
    header_re = re.compile(
        rf"^(?:(?:Automated\s+)?rollback\s+of\s+commit\s+{re.escape(short_prefix)}[0-9a-f]*|This\s+reverts\s+commit\s+{re.escape(short_prefix)}[0-9a-f]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    for rev_sha in git_log_grep_shas(f"{commit_sha}..{mainline_ref}", grep_pat, ignore_case=True):
        if rev_sha in excluded or rev_sha in visited:
            continue
        header_part = run_git(["show", "-s", "--format=%B", rev_sha]).stdout.split(
            "*** Original change description ***", 1
        )[0]
        if header_re.search(header_part):
            if find_revert_on_master(rev_sha, mainline_ref, excluded, visited, _depth + 1) is None:
                return rev_sha
    return None


def find_mainline_commits(pr_title, pr_body, pr_commits, pr_comments, pr_author, mainline_ref, base_sha):
    """Identifies mainline commits on master referenced by or matching the cherry-pick PR."""
    pr_commit_shas = {str(c.get("sha") or "").strip() for c in pr_commits}
    candidates = []
    seen = set()

    def is_revert_subject(subj_text):
        lower = (subj_text or "").strip().lower()
        return lower.startswith(('revert "', "rollback of ", "automated rollback of "))

    def add_candidate(raw_sha, reason, *, allow_revert=True, require_on_master=False):
        full = resolve_commit_sha(raw_sha)
        if not full or full in seen or full in pr_commit_shas:
            return False
        if base_sha and is_ancestor(full, base_sha):
            return False
        if require_on_master and not is_ancestor(full, mainline_ref):
            return False
        if not allow_revert:
            subj_res = run_git(["show", "-s", "--format=%s", full])
            if subj_res.returncode == 0 and is_revert_subject(subj_res.stdout):
                return False
        seen.add(full)
        candidates.append((full, reason))
        return True

    def add_first_from_log(grep_pat, reason, *, allow_revert=True):
        for sha in git_log_grep_shas(mainline_ref, grep_pat):
            if add_candidate(sha, reason, allow_revert=allow_revert):
                return True
        return False

    def extract_explicit_from_text(text, *, allow_rev_id=True, allow_commit_url=True):
        found = False
        for regex, label in (
            (CHERRY_PICK_TRAILER_RE, "cherry-pick trailer"),
            (BAZEL_IO_COMMIT_TRAILER_RE, "Commit trailer"),
        ):
            for m in regex.finditer(text):
                if add_candidate(m.group(1), f"{label} ({m.group(1)[:10]})"):
                    found = True
        if found:
            return True
        if allow_rev_id:
            unquoted = text.split("*** Original change description ***", 1)[0]
            for m in PIPER_REV_ID_RE.finditer(unquoted):
                rev_id = m.group(1)
                if add_first_from_log(
                    rf"^PiperOrigin-RevId:[[:space:]]*{rev_id}[[:space:]]*$", f"PiperOrigin-RevId: {rev_id}"
                ):
                    found = True
            if found:
                return True
        if allow_commit_url:
            for m in GITHUB_COMMIT_URL_RE.finditer(text):
                if add_candidate(m.group(1), f"GitHub commit URL ({m.group(1)[:10]})"):
                    found = True
        return found

    def extract_fallback_from_text(text, subject_line=""):
        found = False
        allow_revert = is_revert_subject(RELEASE_PREFIX_RE.sub("", subject_line or ""))
        unquoted = (text or "").split("*** Original change description ***", 1)[0]
        for m in CHANGE_ID_RE.finditer(unquoted):
            cid = m.group(1)
            if add_first_from_log(
                rf"^Change-Id:[[:space:]]*{cid}[[:space:]]*$", f"Change-Id: {cid[:16]}...", allow_revert=allow_revert
            ):
                found = True
        if found:
            return True

        filtered_text = "\n".join(
            ln for ln in unquoted.splitlines() if not ln.lstrip().startswith(("COPYBARA_INTEGRATE_REVIEW=", "Baseline:"))
        )
        for m in BARE_FULL_SHA_RE.finditer(filtered_text):
            if add_candidate(m.group(1), f"commit SHA ({m.group(1)[:10]})", require_on_master=True):
                found = True
        if not found and re.search(r"\b(?:cherry[- ]?pick|backport|master|mainline|commit)\b", filtered_text, re.I):
            for m in BACKTICK_SHORT_SHA_RE.finditer(filtered_text):
                if add_candidate(m.group(1), f"commit SHA ({m.group(1)[:10]})", require_on_master=True):
                    found = True
        if found:
            return True

        pr_nums = [m.group(1) for m in (TITLE_PR_RE.search(subject_line or ""),) if m]
        for regex in (COPYBARA_PR_RE, FORKED_FROM_RE, BACKPORT_PR_RE):
            pr_nums.extend(m.group(1) for m in regex.finditer(unquoted))
        for num in pr_nums:
            if add_first_from_log(
                rf"((^|[^0-9])Closes #{num}([^0-9]|$)|\(#{num}\))", f"mainline PR #{num}", allow_revert=allow_revert
            ):
                found = True
        if found:
            return True

        cleaned_subj = TITLE_PR_RE.sub(
            "", RELEASE_PREFIX_RE.sub("", re.sub(r"[\x00-\x1f]+", " ", subject_line or ""))
        ).strip()
        if len(cleaned_subj) >= 12 and cleaned_subj.lower() not in GENERIC_COMMIT_SUBJECTS:
            for sha in git_log_grep_shas(mainline_ref, cleaned_subj, fixed=True):
                actual = TITLE_PR_RE.sub(
                    "", RELEASE_PREFIX_RE.sub("", run_git(["show", "-s", "--format=%s", sha]).stdout.strip())
                ).strip()
                if actual == cleaned_subj and add_candidate(
                    sha, "commit subject match on master", allow_revert=allow_revert
                ):
                    return True
        return False

    for c in pr_commits:
        msg = _commit_message(c)
        first_line = msg.splitlines()[0].strip() if msg.strip() else ""
        if not extract_explicit_from_text(msg, allow_rev_id=True, allow_commit_url=True):
            extract_fallback_from_text(msg, subject_line=first_line)

    has_commit_matches = bool(candidates)
    body_matched = extract_explicit_from_text(
        pr_body or "", allow_rev_id=True, allow_commit_url=not has_commit_matches
    )
    for c in pr_comments:
        if str((c.get("user") or {}).get("login") or "").lower() == pr_author.lower():
            extract_explicit_from_text(
                str(c.get("body") or ""), allow_rev_id=False, allow_commit_url=not (has_commit_matches or body_matched)
            )

    if not candidates and not body_matched:
        extract_fallback_from_text(pr_body or "", subject_line=pr_title or "")

    return candidates


def extract_breaking_change_signals(
    mainline_commits_info,
    pr_title,
    pr_body,
    base_sha="",
    head_sha="",
    changed_files=None,
    pr_commits=None,
    java_patches=None,
):
    """Detects potential breaking / backwards-incompatible change indicators."""
    signals = []
    seen = set()

    def add_signal(sig):
        if sig not in seen:
            seen.add(sig)
            signals.append(sig)

    texts = [("PR description", pr_body or ""), ("PR title", pr_title or "")]
    for mc in mainline_commits_info:
        texts.append((f"mainline commit {mc['sha'][:10]}", mc.get("raw_message") or ""))
    for c in pr_commits or []:
        c_msg = _commit_message(c)
        if c_msg:
            texts.append((f"PR commit {str(c.get('sha') or '')[:10] or 'commit'}", c_msg))

    for source_label, text in texts:
        cleaned_text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        for m in re.finditer(r"RELNOTES\[INC\]:\s*(.+)", cleaned_text):
            note = sanitize_untrusted_str(m.group(1), max_len=200)
            if note and note.lower() not in ("none", "none.", "n/a"):
                add_signal(f"Explicit breaking release note (`RELNOTES[INC]`) in {source_label}: {note}")

        m_api = re.search(
            r"###\s*Build API Changes\s*\n+(.+?)(?:\n###|\nPiperOrigin-RevId:|\nCloses #|\Z)",
            cleaned_text,
            re.DOTALL,
        )
        if m_api:
            cleaned_api = m_api.group(1).strip()
            if cleaned_api and cleaned_api.lower() not in ("no", "none", "no.", "none.", "n/a"):
                add_signal(
                    f"Non-empty 'Build API Changes' declared in {source_label}: "
                    f"{sanitize_untrusted_str(cleaned_api, max_len=200)}"
                )

        first_line = cleaned_text.splitlines()[0].strip() if cleaned_text.strip() else ""
        if first_line and FLIP_INCOMPATIBLE_RE.search(first_line):
            add_signal(f"Incompatible flag flip mentioned in {source_label}: {sanitize_untrusted_str(first_line, 200)}")

    if java_patches is None:
        java_paths = [
            f["path"]
            for f in (changed_files or [])
            if isinstance(f, dict) and not f.get("unsafe_path") and str(f.get("path") or "").endswith(".java")
        ]
        java_patches = {
            p: run_git(["--literal-pathspecs", "diff", "-M", base_sha, head_sha, "--", p]).stdout
            for p in java_paths
            if SHA_RE.fullmatch(base_sha) and SHA_RE.fullmatch(head_sha)
        }

    for path, patch_text in sorted(java_patches.items()):
        if not patch_text or not path.endswith(".java"):
            continue
        for hunk in re.split(r"(?m)^@@[^\n]*@@\n?", patch_text):
            if not hunk:
                continue
            if "OptionMetadataTag.INCOMPATIBLE_CHANGE" in hunk or 'name = "incompatible_' in hunk:
                has_removed_default = bool(re.search(r"^-\s*defaultValue\s*=", hunk, re.MULTILINE))
                has_added_default = bool(re.search(r"^\+\s*defaultValue\s*=", hunk, re.MULTILINE))
                has_added_true_default = bool(re.search(r'^\+\s*defaultValue\s*=\s*"true"', hunk, re.MULTILINE))
                if (has_removed_default and has_added_default) or (not has_removed_default and has_added_true_default):
                    add_signal(
                        f"Flips or sets `defaultValue` on an incompatible flag in `{sanitize_untrusted_str(path, 200)}`."
                    )
                    break
        if re.search(r"^-\s*@StarlarkMethod", patch_text, re.MULTILINE) and not re.search(
            r"^\+\s*@StarlarkMethod", patch_text, re.MULTILINE
        ):
            add_signal(
                f"Removes `@StarlarkMethod` annotation in `{sanitize_untrusted_str(path, 200)}` (potential Starlark API break)."
            )

    return signals


def stage_pr_and_base_files(base_sha, head_sha, stage_dir=pathlib.Path("gemini-artifacts/pr")):
    """Stages PR unified diff, head files, and base files via `git diff --numstat`, `git diff`, and `git archive`."""
    head_root = stage_dir / "head"
    base_root = stage_dir / "base"
    head_root.mkdir(parents=True, exist_ok=True)
    base_root.mkdir(parents=True, exist_ok=True)
    diff_path = stage_dir / "diff.patch"

    records = iter_diff_tree_records(base_sha, head_sha, detect_renames=True) or []
    numstat = get_numstat_map(base_sha, head_sha)
    status_map = {"A": "added", "D": "removed", "R": "renamed"}

    base_paths_to_stage = []
    head_paths_to_stage = []
    diff_pathspecs = []
    omitted_notes = []
    changed_files_info = []
    pending_lookup = []

    for idx, (raw_fn, raw_prev, old_mode, new_mode, _, _, st_letter) in enumerate(records):
        status = status_map.get(st_letter, "modified")
        is_safe = is_safe_repo_rel_path(raw_fn) and (not raw_prev or is_safe_repo_rel_path(raw_prev))
        record_path = raw_fn if is_safe_repo_rel_path(raw_fn) else sanitize_untrusted_str(raw_fn, max_len=200)
        adds, dels = numstat.get(raw_fn, (0, 0)) if is_safe else (0, 0)

        base_lookup = (raw_prev if (status == "renamed" and raw_prev) else raw_fn) if is_safe else None
        if idx < 100 and is_safe:
            if status != "added" and old_mode in ("100644", "100755"):
                base_paths_to_stage.append(base_lookup)
            if status != "removed" and new_mode in ("100644", "100755"):
                head_paths_to_stage.append(raw_fn)

        if not is_safe:
            omitted_notes.append(f"# Unsafe path omitted from diff.patch: {record_path}")
        elif raw_fn in LOCKFILE_PATHS and (adds + dels) > MAX_LOCKFILE_DIFF_LINES:
            omitted_notes.append(f"# Large lockfile diff omitted from diff.patch: {raw_fn} (+{adds}/-{dels})")
        else:
            diff_pathspecs.append(raw_fn)
            if raw_prev and raw_prev != raw_fn:
                diff_pathspecs.append(raw_prev)

        pending_lookup.append((record_path, status, adds, dels, is_safe, raw_fn if is_safe else None, base_lookup))

    staged_base = extract_git_files_to_dir(base_sha, base_paths_to_stage, base_root)
    staged_head = extract_git_files_to_dir(head_sha, head_paths_to_stage, head_root)

    for record_path, status, adds, dels, is_safe, head_key, base_key in pending_lookup:
        entry = {
            "path": record_path,
            "status": status,
            "additions": adds,
            "deletions": dels,
            "head_file_path": staged_head.get(head_key) if head_key else None,
            "base_file_path": staged_base.get(base_key) if base_key else None,
        }
        if not is_safe:
            entry["unsafe_path"] = True
        changed_files_info.append(entry)

    diff_body = ""
    if diff_pathspecs:
        unique_specs = list(dict.fromkeys(diff_pathspecs))
        diff_body = run_git(["--literal-pathspecs", "diff", "-M", base_sha, head_sha, "--", *unique_specs]).stdout
    if omitted_notes:
        diff_body = (diff_body.rstrip("\n") + "\n\n" if diff_body else "") + "\n".join(omitted_notes) + "\n"
    diff_path.write_text(diff_body, encoding="utf-8")

    return (
        str(diff_path.resolve()),
        str(head_root.resolve()),
        str(base_root.resolve()),
        changed_files_info,
    )


def simulate_cherry_pick_tree(current_tree, commit_sha, parents):
    """Simulates cherry-picking `commit_sha` onto `current_tree` in memory via `git merge-tree --write-tree`."""
    base_flag = f"--merge-base={parents[0]}" if parents else "--allow-unrelated-histories"
    res = run_git(["merge-tree", "--write-tree", "--name-only", "-z", base_flag, current_tree, commit_sha])
    if res.returncode not in (0, 1) or not res.stdout:
        return None, False, [], {}, ""

    parts = res.stdout.split("\0")
    result_tree = parts[0].strip()
    if not SHA_RE.fullmatch(result_tree):
        return None, False, [], {}, ""
    if res.returncode == 0:
        return result_tree, True, [], {}, ""

    sep = parts.index("", 1) if "" in parts[1:] else len(parts)
    conflicted_files = sorted({p for p in parts[1:sep] if is_safe_repo_rel_path(p)})
    conflict_notes = [chunk.strip() for chunk in parts[sep + 1 :] if "CONFLICT (" in chunk]

    per_file_conflicts = {}
    diff_sections = []
    if conflict_notes:
        diff_sections.append("\n".join(f"# {line}" for note in conflict_notes for line in note.splitlines()))
    for fp in conflicted_files:
        fp_diff = run_git(["--literal-pathspecs", "diff", "--no-renames", current_tree, result_tree, "--", fp]).stdout
        if fp_diff.strip():
            per_file_conflicts[fp] = fp_diff.rstrip()
            diff_sections.append(fp_diff.rstrip())
        elif conflict_notes:
            matching = [n for n in conflict_notes if fp in n] or conflict_notes
            per_file_conflicts[fp] = "\n".join(matching)

    full_conflict_text = ("\n\n".join(diff_sections) + "\n") if diff_sections else ""
    return result_tree, False, conflicted_files, per_file_conflicts, full_conflict_text


def _inspect_mainline_commits(candidates, mainline_ref, base_sha, mainline_dir):
    """Topologically orders `candidates`, inspects patches, and simulates cherry-picking onto `base_sha`."""
    ordered = [(s, by) for s, by in candidates if SHA_RE.fullmatch(s)]
    if len(ordered) > 1:
        rev_range = f"{base_sha}..{mainline_ref}" if (base_sha and SHA_RE.fullmatch(base_sha)) else mainline_ref
        topo_shas = run_git(["rev-list", "--topo-order", "--reverse", rev_range]).stdout.splitlines()
        topo_rank = {sha.strip(): idx for idx, sha in enumerate(topo_shas)}
        ordered.sort(key=lambda item: topo_rank.get(item[0], len(topo_rank)))

    candidate_sha_set = {s for s, _ in ordered}
    current_tree = base_sha if (SHA_RE.fullmatch(base_sha) and resolve_commit_sha(base_sha)) else None
    commits_info = []
    mainline_file_commits = {}
    conflicted_set = set()
    conflict_diffs = {}

    for full_sha, matched_by in ordered:
        on_master = is_ancestor(full_sha, mainline_ref)
        parents = get_commit_parents(full_sha)
        first_parent = parents[0] if parents else EMPTY_TREE_SHA
        meta_parts = run_git(["show", "-s", "--format=%s%x00%an <%ae>%x00%ci%x00%B", full_sha]).stdout.split("\0", 3)
        subj, author, date, raw_message = (
            (p.strip() for p in meta_parts) if len(meta_parts) == 4 else ("", "", "", "")
        )
        show_args = (
            ["show", "-m", "--first-parent", "--stat", "-p", full_sha]
            if len(parents) > 1
            else ["show", "--stat", "-p", full_sha]
        )
        patch_file = mainline_dir / f"{full_sha}.patch"
        patch_file.write_text(run_git(show_args).stdout, encoding="utf-8")

        commit_entries = get_diff_tree_entries(first_parent, full_sha) or {}
        files_modified = sorted(p for p in commit_entries if is_safe_repo_rel_path(p))

        applied_cleanly = None
        conflicted_files = []
        conflict_diff_file_str = None
        if on_master:
            for fp in files_modified:
                mainline_file_commits.setdefault(fp, []).append((first_parent, full_sha))
            if current_tree is not None:
                current_tree, applied_cleanly, conflicted_files, per_file_c, full_c_text = simulate_cherry_pick_tree(
                    current_tree, full_sha, parents
                )
                if not applied_cleanly:
                    conflicted_set.update(conflicted_files)
                    conflict_diffs.update(per_file_c)
                    if full_c_text:
                        c_file = mainline_dir / f"{full_sha}_conflicts.diff"
                        c_file.write_text(full_c_text, encoding="utf-8")
                        conflict_diff_file_str = str(c_file.resolve())

        commits_info.append(
            {
                "sha": full_sha,
                "subject": sanitize_untrusted_str(subj, max_len=200),
                "author": sanitize_untrusted_str(author, max_len=120),
                "date": sanitize_untrusted_str(date, max_len=60),
                "matched_by": matched_by,
                "is_on_master": on_master,
                "reverted_by_on_master": (
                    find_revert_on_master(full_sha, mainline_ref, excluded_shas=candidate_sha_set)
                    if on_master
                    else None
                ),
                "files_modified": files_modified,
                "patch_file": str(patch_file.resolve()),
                "applied_cleanly_on_base": applied_cleanly,
                "conflicted_files": conflicted_files,
                "conflict_diff_file": conflict_diff_file_str,
                "raw_message": raw_message,
            }
        )

    return commits_info, mainline_file_commits, current_tree, conflicted_set, conflict_diffs


def analyze_mainline_and_compare(
    candidates,
    mainline_ref,
    base_sha,
    head_sha,
    base_root_path,
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
):
    """Inspects mainline commits, simulates cherry-picking onto `base_sha`, and compares Git trees."""
    mainline_dir = stage_dir / "mainline"
    mainline_dir.mkdir(parents=True, exist_ok=True)
    base_root = pathlib.Path(base_root_path)

    (
        mainline_commits_info,
        mainline_file_commits,
        current_tree,
        all_conflicted_files,
        conflict_diffs_by_file,
    ) = _inspect_mainline_commits(candidates, mainline_ref, base_sha, mainline_dir)

    pr_diff_entries = get_diff_tree_entries(base_sha, head_sha) or {}
    files_truncated = len(pr_diff_entries) > MAX_AUTO_CLEAN_FILES
    cp_diff_entries = get_diff_tree_entries(base_sha, current_tree) if current_tree is not None else None

    pr_file_set = {p for p in pr_diff_entries if is_safe_repo_rel_path(p)}
    unsafe_paths = sorted(p for p in pr_diff_entries if not is_safe_repo_rel_path(p))
    mainline_file_set = set(mainline_file_commits.keys())
    all_files = sorted(pr_file_set | mainline_file_set)

    omitted_regular_files = [
        fp
        for fp in (mainline_file_set - pr_file_set)
        if (cp_diff_entries or {}).get(fp, ("",))[0] in ("100644", "100755")
    ]
    if omitted_regular_files:
        extract_git_files_to_dir(base_sha, omitted_regular_files, base_root)

    verified_mainline = [mc for mc in mainline_commits_info if mc["is_on_master"]]
    has_unverified_mainline = any(not mc["is_on_master"] for mc in mainline_commits_info)
    last_mainline_sha = verified_mainline[-1]["sha"] if verified_mainline else None
    file_comparisons = []
    vs_mainline_sections = []

    def format_file_section(fp, classification, *, in_pr, in_mainline):
        is_lockfile_regen = classification == "lockfile_regenerated"
        sec = ["=" * 80, f"FILE: {fp}", f"CLASSIFICATION: {classification}", "=" * 80]

        if not is_lockfile_regen and base_sha and last_mainline_sha:
            hist_res = run_git(
                ["--literal-pathspecs", "log", "--oneline", "-n", "15", f"{base_sha}..{last_mainline_sha}", "--", fp]
            )
            if hist_res.returncode == 0 and hist_res.stdout.strip():
                sec.append(f"\n--- Commits touching {fp} on master ({base_sha[:10]}..{last_mainline_sha[:10]}) ---")
                sec.append(hist_res.stdout.strip())

        if in_mainline and not is_lockfile_regen:
            ml_patches = [
                run_git(["--literal-pathspecs", "diff", "--no-renames", p_sha, c_sha, "--", fp]).stdout.rstrip()
                for p_sha, c_sha in mainline_file_commits.get(fp, ())
            ]
            sec.append(f"\n--- Mainline commit patch for {fp} ---")
            sec.append("\n".join(p for p in ml_patches if p) or "(empty patch)")

        if fp in conflict_diffs_by_file and not is_lockfile_regen:
            sec.append(f"\n--- 3-way git cherry-pick conflict markers on {base_sha[:10]} for {fp} ---")
            sec.append(conflict_diffs_by_file[fp])

        if in_pr:
            pr_patch = run_git(["--literal-pathspecs", "diff", "--no-renames", base_sha, head_sha, "--", fp]).stdout.rstrip()
            sec.append(f"\n--- PR patch for {fp} ---")
            if is_lockfile_regen and pr_patch:
                patch_lines = pr_patch.splitlines()
                sec.append(
                    "\n".join(patch_lines[:200]) + "\n... (lockfile patch truncated; see pr_diff_path)"
                    if len(patch_lines) > 200
                    else pr_patch
                )
            else:
                sec.append(pr_patch or "(empty patch)")

        if not is_lockfile_regen and current_tree is not None and fp not in all_conflicted_files:
            deviation = run_git(
                [
                    "--literal-pathspecs",
                    "diff",
                    "--no-renames",
                    "--src-prefix=clean_cherry_pick/",
                    "--dst-prefix=pr_head/",
                    current_tree,
                    head_sha,
                    "--",
                    fp,
                ]
            ).stdout.rstrip()
            if deviation:
                sec.append(
                    f"\n--- Exact deviation between clean cherry-pick onto release branch and PR head for {fp} ---"
                )
                sec.append(deviation)

        return "\n".join(sec)

    for fp in all_files:
        in_pr = fp in pr_file_set
        in_mainline = fp in mainline_file_set
        matches_clean_cp = (
            cp_diff_entries is not None
            and fp not in all_conflicted_files
            and pr_diff_entries.get(fp) == cp_diff_entries.get(fp)
        )
        if in_pr and in_mainline:
            if matches_clean_cp:
                classification = "identical"
            elif fp in LOCKFILE_PATHS:
                classification = "lockfile_regenerated"
            elif fp in all_conflicted_files:
                classification = "adapted_with_conflict"
            else:
                classification = "adapted_without_conflict"
        elif in_pr:
            classification = "lockfile_regenerated" if fp in LOCKFILE_PATHS else "extra_in_pr"
        else:
            classification = "identical" if matches_clean_cp else "omitted_from_pr"

        file_comparisons.append(
            {
                "path": fp,
                "classification": classification,
                "in_pr": in_pr,
                "in_mainline": in_mainline,
                "had_merge_conflict": fp in all_conflicted_files,
            }
        )
        if classification != "identical":
            vs_mainline_sections.append(
                format_file_section(fp, classification, in_pr=in_pr, in_mainline=in_mainline)
            )

    for raw_uf in unsafe_paths:
        safe_uf_path = sanitize_untrusted_str(raw_uf, max_len=200)
        file_comparisons.append(
            {
                "path": safe_uf_path,
                "classification": "extra_in_pr",
                "in_pr": True,
                "in_mainline": False,
                "had_merge_conflict": False,
                "unsafe_path": True,
            }
        )
        vs_mainline_sections.append(
            f"{'=' * 80}\nFILE: {safe_uf_path}\n"
            f"CLASSIFICATION: extra_in_pr (rejected by repository-relative path validator)\n{'=' * 80}"
        )

    def add_note(title, detail=""):
        lines = ["=" * 80, f"NOTE: {title}"]
        if detail:
            lines.append(detail)
        lines.append("=" * 80)
        vs_mainline_sections.append("\n".join(lines))

    reverted_pairs = [
        f"{mc['sha'][:12]} (reverted by {mc['reverted_by_on_master'][:12]})"
        for mc in verified_mainline
        if mc.get("reverted_by_on_master")
    ]
    if reverted_pairs:
        add_note(
            "MAINLINE COMMIT WAS REVERTED OR ROLLED BACK ON MASTER",
            f"The following mainline commit(s) were later reverted/rolled back on {mainline_ref}: {', '.join(reverted_pairs)}",
        )
    if has_unverified_mainline:
        unverified_shas = ", ".join(mc["sha"][:12] for mc in mainline_commits_info if not mc["is_on_master"])
        add_note(
            "UNVERIFIED COMMIT(S) REFERENCED BY PR",
            f"The following referenced commit(s) are not ancestors of {mainline_ref}: {unverified_shas}",
        )
    if files_truncated:
        add_note(
            "PR FILE LIST TRUNCATED",
            "The pull request touches too many files for automatic CLEAN classification.",
        )
    if verified_mainline and not pr_diff_entries and not vs_mainline_sections:
        add_note("PULL REQUEST HAS NO CHANGED FILES")

    if not verified_mainline:
        overall_classification = "NO_MAINLINE_COMMIT"
    elif has_unverified_mainline or files_truncated or unsafe_paths or not pr_diff_entries:
        overall_classification = "MODIFIED_OR_EXTRA_CHANGES"
    else:
        non_clean = [fc for fc in file_comparisons if fc["classification"] not in ("identical", "lockfile_regenerated")]
        if not non_clean:
            overall_classification = "CLEAN"
        elif all(fc["classification"] == "adapted_with_conflict" for fc in non_clean):
            overall_classification = "CONFLICT_RESOLVED"
        else:
            overall_classification = "MODIFIED_OR_EXTRA_CHANGES"

    vs_mainline_diff_path = None
    if vs_mainline_sections:
        vs_path = stage_dir / "vs_mainline.diff"
        vs_path.write_text("\n\n".join(vs_mainline_sections) + "\n", encoding="utf-8")
        vs_mainline_diff_path = str(vs_path.resolve())

    return (
        mainline_commits_info,
        file_comparisons,
        overall_classification,
        vs_mainline_diff_path,
        files_truncated,
    )


def is_authentic_gemini_review(review_obj, pr_author=""):
    """Returns True if `review_obj` is a genuine automated Gemini review rather than a quote or spoof."""
    if not isinstance(review_obj, dict):
        return False
    state = str(review_obj.get("state") or "").upper()
    if state not in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED"):
        return False
    body = str(review_obj.get("body") or "").lstrip()
    has_self_review_banner = body.startswith(AUTOMATED_DECISION_PREFIX)
    if not (
        body.startswith((f"## 📋 {GEMINI_REVIEW_MARKER}", f"## {GEMINI_REVIEW_MARKER}", GUARDRAIL_BANNER_PREFIX))
        or has_self_review_banner
    ):
        return False

    user_info = review_obj.get("user")
    login = str((user_info or {}).get("login") or "").strip().lower() if isinstance(user_info, dict) else ""
    if not login:
        return False
    if login in {"bazel-io", "github-actions[bot]", "gemini-code-assist[bot]"}:
        return True
    if pr_author and login == pr_author.strip().lower() and not has_self_review_banner and state == "COMMENTED":
        return False
    return str(review_obj.get("author_association") or "").strip().upper() in TRUSTED_AUTHOR_ASSOCIATIONS


def has_active_gemini_approval(pr_reviews, pr_author=""):
    """Returns True if an authentic Gemini approval has not been superseded by CHANGES_REQUESTED."""
    active = False
    for r in pr_reviews:
        if not is_authentic_gemini_review(r, pr_author=pr_author):
            continue
        st = effective_review_state(r)
        if st in ("APPROVED", "CHANGES_REQUESTED"):
            active = st == "APPROVED"
    return active


def _git_patch_id(from_sha, to_sha):
    diff_res = run_git_bytes(["diff", "--binary", from_sha, to_sha])
    if diff_res.returncode != 0:
        return None
    if not diff_res.stdout:
        return "empty"
    pid_res = subprocess.run(
        ["git", "patch-id", "--stable"], input=diff_res.stdout, capture_output=True, check=False
    )
    return pid_res.stdout.decode("utf-8", errors="replace").split()[0] if pid_res.returncode == 0 and pid_res.stdout else None


def _compute_since_last_review_delta(prev_head_sha, base_ref, base_sha, head_sha, stage_dir):
    """Computes `(pr_patch_unchanged, since_last_review_diff_path)` via `git merge-tree`, `diff`, and `range-diff`."""
    if not SHA_RE.fullmatch(prev_head_sha):
        return None, None
    stage_dir.mkdir(parents=True, exist_ok=True)
    since_path = stage_dir / "since_last_review.diff"
    if prev_head_sha == head_sha:
        since_path.write_text(
            f"No changes since last review: commit {head_sha} was already reviewed.\n", encoding="utf-8"
        )
        return True, str(since_path.resolve())

    if not resolve_commit_sha(prev_head_sha):
        run_git(["fetch", "--filter=blob:none", "--no-tags", "origin", prev_head_sha])
    if not (resolve_commit_sha(prev_head_sha) and SHA_RE.fullmatch(base_sha) and resolve_commit_sha(head_sha)):
        return None, None

    base_branch_ref = f"origin/{base_ref}"
    if run_git(["rev-parse", "--verify", f"{base_branch_ref}^{{commit}}"]).returncode != 0:
        base_branch_ref = base_sha
    mb_res = run_git(["merge-base", base_branch_ref, prev_head_sha])
    prev_base_sha = mb_res.stdout.strip() if mb_res.returncode == 0 else base_sha
    if not SHA_RE.fullmatch(prev_base_sha):
        return None, None

    if prev_base_sha == base_sha:
        baseline_tree, rebased_cleanly = prev_head_sha, True
    else:
        baseline_tree, rebased_cleanly, _, _, _ = simulate_cherry_pick_tree(base_sha, prev_head_sha, [prev_base_sha])

    if rebased_cleanly and baseline_tree is not None:
        tree_delta = get_diff_tree_entries(baseline_tree, head_sha) or {}
        if not tree_delta:
            since_path.write_text(
                f"No effective PR patch changes between last reviewed commit {prev_head_sha} "
                f"and current head {head_sha} (pure rebase or identical diff).\n",
                encoding="utf-8",
            )
            return True, str(since_path.resolve())
        diff_out = run_git(["diff", "-M", baseline_tree, head_sha]).stdout
        since_path.write_text(diff_out or f"Binary or metadata changes in: {', '.join(sorted(tree_delta))}\n", encoding="utf-8")
        return False, str(since_path.resolve())

    old_pid = _git_patch_id(prev_base_sha, prev_head_sha)
    new_pid = _git_patch_id(base_sha, head_sha)
    unchanged = bool(old_pid and old_pid == new_pid)
    range_diff = run_git(["range-diff", f"{prev_base_sha}..{prev_head_sha}", f"{base_sha}..{head_sha}"]).stdout
    since_path.write_text(range_diff or "Rebased across conflicting upstream changes.\n", encoding="utf-8")
    return unchanged, str(since_path.resolve())


def analyze_prior_review_delta(
    pr_reviews,
    pr_inline_comments,
    base_ref,
    base_sha,
    head_sha,
    pr_author="",
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
):
    """Summarizes prior Gemini reviews/inline comments and computes delta since the last reviewed commit."""
    gemini_reviews = [
        {
            "id": r.get("id"),
            "state": sanitize_untrusted_str(str(r.get("state") or "").upper(), max_len=40),
            "commit_id": sanitize_untrusted_str(r.get("commit_id") or "", max_len=40),
            "submitted_at": sanitize_untrusted_str(r.get("submitted_at") or "", max_len=40),
            "body": sanitize_untrusted_multiline(r.get("body") or "", max_len=2000),
        }
        for r in pr_reviews
        if is_authentic_gemini_review(r, pr_author=pr_author)
    ]

    serialized_inline_comments = []
    for c in pr_inline_comments[-30:]:
        if not isinstance(c, dict):
            continue
        raw_c_path = str(c.get("path") or "")
        serialized_inline_comments.append(
            {
                "id": c.get("id"),
                "author": sanitize_untrusted_str((c.get("user") or {}).get("login") or "", max_len=80),
                "path": raw_c_path if is_safe_repo_rel_path(raw_c_path) else sanitize_untrusted_str(raw_c_path, 200),
                "line": c.get("line") or c.get("original_line"),
                "in_reply_to_id": c.get("in_reply_to_id"),
                "created_at": sanitize_untrusted_str(c.get("created_at") or "", max_len=40),
                "body": sanitize_untrusted_multiline(c.get("body") or "", max_len=1000),
            }
        )

    last_review = gemini_reviews[-1] if gemini_reviews else None
    pr_patch_unchanged, since_path = (
        _compute_since_last_review_delta(
            str(last_review.get("commit_id") or "").strip(), base_ref, base_sha, head_sha, stage_dir
        )
        if last_review is not None
        else (None, None)
    )
    return bool(gemini_reviews), last_review, pr_patch_unchanged, since_path, serialized_inline_comments


def render_review_prompt(
    context_json_str,
    template_path=pathlib.Path(".github/actions/cherry-pick-review/prompt_template.md"),
    out_path=pathlib.Path(".gemini/prompt.md"),
):
    """Renders .gemini/prompt.md by expanding context.json into prompt_template.md."""
    if not template_path.is_file():
        fallback_tpl = pathlib.Path(__file__).resolve().parent / "prompt_template.md"
        if fallback_tpl.is_file():
            template_path = fallback_tpl
    if not template_path.is_file():
        return False
    rendered = template_path.read_text(encoding="utf-8").replace("@{.gemini/context.json}", context_json_str.strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return True


def main():
    repo = os.environ["REPOSITORY"].strip()
    event_name = os.environ["EVENT_NAME"].strip()
    event_action = os.environ.get("EVENT_ACTION", "").strip()
    pr_number = str(os.environ["PR_NUMBER"]).strip()
    comment_body = os.environ.get("COMMENT_BODY", "").strip()
    input_context = os.environ.get("INPUT_CONTEXT", "").strip()
    github_output = os.environ.get("GITHUB_OUTPUT", "/dev/null")

    if not REPO_RE.fullmatch(repo):
        raise ValueError(f"Invalid repository identifier: {repo!r}")
    if not PR_NUMBER_RE.fullmatch(pr_number):
        raise ValueError(f"Invalid pull request number: {pr_number!r}")

    def skip_run(log_msg, summary_msg):
        print(log_msg)
        append_step_summary([f"- **Outcome:** ⏭️ {summary_msg}"])
        with open(github_output, "a", encoding="utf-8") as out:
            out.write("should_run=false\n")
        return 0

    pr_info = gh_api(f"repos/{repo}/pulls/{pr_number}")
    pr_state = pr_info.get("state")
    is_draft = bool(pr_info.get("draft"))
    base_ref = str((pr_info.get("base") or {}).get("ref") or "").strip()
    base_sha = str((pr_info.get("base") or {}).get("sha") or "").strip()
    head_sha = str((pr_info.get("head") or {}).get("sha") or "").strip()

    is_manual = event_name in ("workflow_dispatch", "issue_comment") or (
        event_name == "pull_request_target" and event_action == "labeled"
    )
    trigger_type = "manual" if is_manual else "automatic"

    release_version = is_valid_release_branch(base_ref)
    if not release_version:
        safe_base_ref = sanitize_untrusted_str(base_ref, max_len=80)
        return skip_run(
            f"Skipping PR #{pr_number}: base branch {safe_base_ref!r} is not a valid release-* branch.",
            f"Skipped PR #{pr_number} (base branch `{safe_base_ref}` is not a `release-*` branch)",
        )
    if pr_state != "open" or (is_draft and not is_manual):
        return skip_run(
            f"Skipping PR #{pr_number} (state={pr_state}, draft={is_draft}, trigger={trigger_type})",
            f"Skipped PR #{pr_number} (`state={pr_state}`, `draft={is_draft}`, `trigger={trigger_type}`)",
        )

    if not SHA_RE.fullmatch(head_sha):
        raise ValueError(f"Invalid head SHA from GitHub PR API: {head_sha!r}")
    if not SHA_RE.fullmatch(base_sha):
        raise ValueError(f"Invalid base SHA from GitHub PR API: {base_sha!r}")

    raw_context = input_context or re.sub(r"^@gemini-cli\s+/review\b\s*", "", comment_body).strip()
    additional_context = sanitize_untrusted_str(raw_context, max_len=500)

    raw_pr_author = str((pr_info.get("user") or {}).get("login") or "")
    raw_pr_body = str(pr_info.get("body") or "")
    pr_comments = gh_api(f"repos/{repo}/issues/{pr_number}/comments?per_page=100", paginate=True, optional=True)
    pr_reviews = gh_api(f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100", paginate=True, optional=True)
    pr_inline_comments = gh_api(f"repos/{repo}/pulls/{pr_number}/comments?per_page=100", paginate=True, optional=True)

    if run_git(["remote", "get-url", "origin"]).returncode == 0:
        master_fetch = run_git(
            ["fetch", "--filter=blob:none", "--no-tags", "origin", "+refs/heads/master:refs/remotes/origin/master"]
        )
        if master_fetch.returncode != 0:
            run_git(
                ["fetch", "--filter=blob:none", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"],
                check=True,
            )
        run_git(
            ["fetch", "--filter=blob:none", "--no-tags", "origin", f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}"],
            check=True,
        )
        run_git(["fetch", "--filter=blob:none", "--no-tags", "origin", f"refs/pull/{pr_number}/head"], check=True)

    mainline_ref = detect_mainline_ref()
    if not resolve_commit_sha(head_sha):
        raise RuntimeError(f"Could not resolve PR head commit {head_sha} in local git repository")
    mb_res = run_git(["merge-base", base_sha, head_sha])
    effective_base_sha = (
        mb_res.stdout.strip() if mb_res.returncode == 0 and SHA_RE.fullmatch(mb_res.stdout.strip()) else base_sha
    )
    pr_commits = list_git_commits(effective_base_sha, head_sha)

    pr_diff_path, pr_head_root, pr_base_root, changed_files_info = stage_pr_and_base_files(
        base_sha=effective_base_sha,
        head_sha=head_sha,
    )
    candidates = find_mainline_commits(
        pr_title=pr_info.get("title") or "",
        pr_body=raw_pr_body,
        pr_commits=pr_commits,
        pr_comments=pr_comments,
        pr_author=raw_pr_author,
        mainline_ref=mainline_ref,
        base_sha=effective_base_sha,
    )
    (
        mainline_commits_info,
        file_comparisons,
        overall_classification,
        vs_mainline_diff_path,
        files_truncated,
    ) = analyze_mainline_and_compare(
        candidates=candidates,
        mainline_ref=mainline_ref,
        base_sha=effective_base_sha,
        head_sha=head_sha,
        base_root_path=pr_base_root,
    )
    (
        has_existing_gemini_review,
        last_gemini_review,
        pr_patch_unchanged_since_last_review,
        since_last_review_diff_path,
        existing_inline_comments,
    ) = analyze_prior_review_delta(
        pr_reviews=pr_reviews,
        pr_inline_comments=pr_inline_comments,
        base_ref=base_ref,
        base_sha=effective_base_sha,
        head_sha=head_sha,
        pr_author=raw_pr_author,
    )
    breaking_change_signals = extract_breaking_change_signals(
        mainline_commits_info=mainline_commits_info,
        pr_title=pr_info.get("title") or "",
        pr_body=raw_pr_body,
        base_sha=effective_base_sha,
        head_sha=head_sha,
        changed_files=changed_files_info,
        pr_commits=pr_commits,
    )

    serialized_mainline_commits = [
        {
            **{k: v for k, v in mc.items() if k != "raw_message"},
            "message_preview": sanitize_untrusted_multiline(mc.get("raw_message") or "", max_len=1500),
        }
        for mc in mainline_commits_info
    ]
    serialized_pr_commits = [
        {
            "sha": c["sha"][:40],
            "author": sanitize_untrusted_str(c.get("author") or "", max_len=80),
            "message": sanitize_untrusted_multiline(c.get("message") or "", max_len=1000),
        }
        for c in pr_commits[:20]
    ]
    serialized_pr_comments = [
        {
            "author": sanitize_untrusted_str((c.get("user") or {}).get("login") or "", max_len=80),
            "created_at": sanitize_untrusted_str(c.get("created_at") or "", max_len=40),
            "body": sanitize_untrusted_multiline(c.get("body") or "", max_len=1000),
        }
        for c in pr_comments[-10:]
    ]

    non_lockfile_line_changes = sum(
        f["additions"] + f["deletions"]
        for f in changed_files_info
        if not f.get("unsafe_path") and f["path"] not in LOCKFILE_PATHS
    )

    context_data = {
        "repository": repo,
        "workspace_root": str(pathlib.Path.cwd().resolve()),
        "pull_request_number": int(pr_number),
        "base_ref": base_ref,
        "release_version": release_version,
        "is_patch_release": is_patch_release_version(release_version),
        "base_sha": effective_base_sha,
        "head_sha": head_sha,
        "pr_title": sanitize_untrusted_str(pr_info.get("title") or "", max_len=250),
        "pr_body": sanitize_untrusted_multiline(raw_pr_body, max_len=3000),
        "pr_author": sanitize_untrusted_str(raw_pr_author, max_len=80),
        "pr_author_association": sanitize_untrusted_str(pr_info.get("author_association") or "", max_len=40).upper(),
        "trigger_type": trigger_type,
        "additional_context": additional_context,
        "files_truncated": files_truncated,
        "non_lockfile_line_changes": non_lockfile_line_changes,
        "has_existing_gemini_review": has_existing_gemini_review,
        "has_prior_active_approval": has_active_gemini_approval(pr_reviews, pr_author=raw_pr_author),
        "last_gemini_review": last_gemini_review,
        "pr_patch_unchanged_since_last_review": pr_patch_unchanged_since_last_review,
        "since_last_review_diff_path": since_last_review_diff_path,
        "existing_inline_comments": existing_inline_comments,
        "pr_commits": serialized_pr_commits,
        "pr_comments": serialized_pr_comments,
        "mainline_commits": serialized_mainline_commits,
        "has_reverted_mainline": any(
            bool(mc.get("reverted_by_on_master")) for mc in mainline_commits_info if mc.get("is_on_master")
        ),
        "overall_cherry_pick_classification": overall_classification,
        "has_lockfile_changes": any(
            fc.get("classification") == "lockfile_regenerated" for fc in file_comparisons
        ),
        "file_comparisons": file_comparisons,
        "breaking_change_signals": breaking_change_signals,
        "pr_diff_path": pr_diff_path,
        "vs_mainline_diff_path": vs_mainline_diff_path,
        "pr_head_root": pr_head_root,
        "pr_base_root": pr_base_root,
        "changed_files": changed_files_info,
    }

    os.makedirs(".gemini", exist_ok=True)
    context_json_str = json.dumps(context_data, indent=2)
    pathlib.Path(".gemini/context.json").write_text(context_json_str, encoding="utf-8")
    if not render_review_prompt(context_json_str):
        raise FileNotFoundError("Could not locate prompt_template.md to render .gemini/prompt.md")

    print("Generated .gemini/context.json:")
    print(context_json_str)
    with open(github_output, "a", encoding="utf-8") as out:
        out.write(f"should_run=true\npr_number={pr_number}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
