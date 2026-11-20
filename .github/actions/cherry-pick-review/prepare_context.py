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

import difflib
import json
import os
import pathlib
import re
import subprocess
import sys

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

RELEASE_BRANCH_RE = re.compile(r"^release-([A-Za-z0-9._-]+)$")
MAX_AUTO_CLEAN_FILES = 500
MAX_PATCH_CHARS = 200_000

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


def resolve_path_within(base_dir, rel_path):
    """Returns base_dir / rel_path if it resolves inside base_dir without symlinks, else None."""
    if not is_safe_repo_rel_path(rel_path) or base_dir.is_symlink():
        return None
    base_resolved = base_dir.resolve()
    cur = base_dir
    for part in rel_path.split("/"):
        cur = cur / part
        if cur.is_symlink():
            return None
    resolved = cur.resolve()
    if resolved == base_resolved or not resolved.is_relative_to(base_resolved):
        return None
    return cur


def safe_write_staged_file(base_dir, rel_path, content):
    """Safely writes `content` to `base_dir / rel_path`, guarding against symlinks and path collisions."""
    dest = resolve_path_within(base_dir, rel_path)
    if dest is None:
        return None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.is_dir():
            return None
        dest.write_text(content, encoding="utf-8")
        return str(dest.resolve())
    except OSError:
        return None


def gh_api(endpoint, paginate=False, slurp=False):
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    if slurp:
        cmd.append("--slurp")
    cmd.append(endpoint)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if res.returncode != 0:
        raise RuntimeError(f"gh api {endpoint} failed (exit {res.returncode}): {res.stderr.strip()}")
    return json.loads(res.stdout)


def gh_api_paginated(endpoint, *, optional=False):
    """Fetches a paginated list endpoint via `gh api --paginate --slurp`."""
    try:
        pages = gh_api(endpoint, paginate=True, slurp=True)
        return [item for page in pages for item in page] if isinstance(pages, list) else []
    except Exception:
        if optional:
            return []
        raise


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


def format_mode_sha(mode, obj_sha):
    if not mode or mode == "000000" or obj_sha == "0" * 40:
        return "deleted"
    obj_type = "commit" if mode == "160000" else "blob"
    return f"{obj_type} {obj_sha[:12]} (mode {mode})"


def read_blob_by_mode_and_sha(mode, blob_sha):
    """Reads a non-binary regular file blob (mode 100644/100755) by mode and SHA."""
    if mode not in ("100644", "100755") or not SHA_RE.fullmatch(blob_sha) or blob_sha == "0" * 40:
        return None
    cat_res = run_git_bytes(["cat-file", "blob", blob_sha])
    if cat_res.returncode != 0 or b"\x00" in cat_res.stdout:
        return None
    return cat_res.stdout.decode("utf-8", errors="replace")


def read_git_regular_file_blob(tree_ish, rel_path):
    """Reads a non-binary regular file blob at `tree_ish:rel_path`, rejecting symlinks and binaries."""
    if not (isinstance(tree_ish, str) and SHA_RE.fullmatch(tree_ish) and is_safe_repo_rel_path(rel_path)):
        return None
    ls_res = run_git(["--literal-pathspecs", "ls-tree", "-z", tree_ish, "--", rel_path])
    if ls_res.returncode != 0 or "\t" not in ls_res.stdout:
        return None
    meta, entry_path = ls_res.stdout.split("\0", 1)[0].split("\t", 1)
    parts = meta.split()
    if entry_path != rel_path or len(parts) != 3 or parts[1] != "blob":
        return None
    return read_blob_by_mode_and_sha(parts[0], parts[2])


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


def normalize_patch_changes(patch_text):
    """Normalizes a unified diff patch by stripping headers while preserving context, metadata, and +/- lines."""
    if not isinstance(patch_text, str) or not patch_text.strip():
        return []
    changes = []
    in_hunk = False
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and line.startswith(("+", "-", " ")):
            changes.append(line)
        elif not in_hunk and line.startswith(("+", "-", "old mode ", "new mode ", "Binary files ")):
            changes.append(line)
    return changes


def split_git_diff_by_file(diff_text):
    """Splits a multi-file `git diff` output into `{safe_filename: patch_body}`."""
    result = {}
    if not diff_text:
        return result
    current_file = None
    minus_a_file = None
    current_hunks = []
    in_hunks = False

    def flush():
        if current_file is not None and is_safe_repo_rel_path(current_file):
            result[current_file] = "\n".join(current_hunks)

    for line in diff_text.splitlines():
        if line.startswith(("diff --git ", "diff --cc ", "diff --combined ", "* Unmerged path ")):
            flush()
            m_git = re.match(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$', line)
            m_cc = re.match(r"^diff --(?:cc|combined) (.+)$", line)
            m_unmerged = re.match(r"^\* Unmerged path (.+)$", line)
            candidate = (
                m_git.group(2).strip('"')
                if m_git
                else (m_cc or m_unmerged).group(1).strip().strip('"')
                if (m_cc or m_unmerged)
                else None
            )
            current_file = candidate if is_safe_repo_rel_path(candidate) else None
            minus_a_file = None
            current_hunks = [line] if m_unmerged else []
            in_hunks = bool(m_unmerged)
            continue

        if not in_hunks:
            if line.startswith("rename to "):
                cand = line[len("rename to ") :].strip('"')
                current_file = cand if is_safe_repo_rel_path(cand) else None
            elif line.startswith("--- a/"):
                minus_a_file = line[len("--- a/") :].strip('"')
            elif line.startswith("+++ b/"):
                cand = line[len("+++ b/") :].strip('"')
                current_file = cand if is_safe_repo_rel_path(cand) else None
            elif line == "+++ /dev/null" and minus_a_file:
                current_file = minus_a_file if is_safe_repo_rel_path(minus_a_file) else None

        if current_file is not None:
            if line.startswith("@@"):
                in_hunks = True
            if in_hunks or line.startswith(
                (
                    "old mode ",
                    "new mode ",
                    "new file mode ",
                    "deleted file mode ",
                    "similarity index ",
                    "rename from ",
                    "rename to ",
                    "Binary files ",
                )
            ):
                current_hunks.append(line)

    flush()
    return result


def git_diff_by_file(*git_args):
    """Runs `git <git_args>` and splits the resulting unified diff by safe file path."""
    res = run_git(git_args)
    return split_git_diff_by_file(res.stdout) if res.returncode == 0 else {}


def count_patch_line_changes(patch_text):
    """Returns `(additions, deletions)` counted from unified diff hunks."""
    adds = dels = 0
    in_hunk = False
    for line in (patch_text or "").splitlines():
        if line.startswith("@@"):
            in_hunk = True
        elif in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1
    return adds, dels


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
        for regex, label in ((CHERRY_PICK_TRAILER_RE, "cherry-pick trailer"), (BAZEL_IO_COMMIT_TRAILER_RE, "Commit trailer")):
            for m in regex.finditer(text):
                if add_candidate(m.group(1), f"{label} ({m.group(1)[:10]})"):
                    found = True
        if found:
            return True
        if allow_rev_id:
            unquoted = text.split("*** Original change description ***", 1)[0]
            for m in PIPER_REV_ID_RE.finditer(unquoted):
                rev_id = m.group(1)
                if add_first_from_log(rf"^PiperOrigin-RevId:[[:space:]]*{rev_id}[[:space:]]*$", f"PiperOrigin-RevId: {rev_id}"):
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

        cleaned_subj = TITLE_PR_RE.sub("", RELEASE_PREFIX_RE.sub("", re.sub(r"[\x00-\x1f]+", " ", subject_line or ""))).strip()
        if len(cleaned_subj) >= 12 and cleaned_subj.lower() not in GENERIC_COMMIT_SUBJECTS:
            for sha in git_log_grep_shas(mainline_ref, cleaned_subj, fixed=True):
                actual = TITLE_PR_RE.sub("", RELEASE_PREFIX_RE.sub("", run_git(["show", "-s", "--format=%s", sha]).stdout.strip())).strip()
                if actual == cleaned_subj and add_candidate(sha, "commit subject match on master", allow_revert=allow_revert):
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


def extract_breaking_change_signals(mainline_commits_info, pr_title, pr_body, pr_patches_by_file, pr_commits=None):
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

    for path, patch_text in sorted((pr_patches_by_file or {}).items()):
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
    """Derives PR changed files and diffs directly from local Git `base_sha..head_sha` and stages blobs."""
    head_root = stage_dir / "head"
    base_root = stage_dir / "base"
    head_root.mkdir(parents=True, exist_ok=True)
    base_root.mkdir(parents=True, exist_ok=True)
    diff_path = stage_dir / "diff.patch"

    records = iter_diff_tree_records(base_sha, head_sha, detect_renames=True) or []
    all_patches = git_diff_by_file("diff", "-M", base_sha, head_sha)

    status_map = {"A": "added", "D": "removed", "R": "renamed"}
    diff_chunks = []
    changed_files_info = []
    pr_patches_by_file = {}

    for idx, (raw_fn, raw_prev, old_mode, new_mode, old_sha, new_sha, st_letter) in enumerate(records):
        status = status_map.get(st_letter, "modified")
        is_safe_fn = is_safe_repo_rel_path(raw_fn)
        is_safe_prev = not raw_prev or is_safe_repo_rel_path(raw_prev)
        is_safe_path = is_safe_fn and is_safe_prev

        record_path = raw_fn if is_safe_fn else sanitize_untrusted_str(raw_fn, max_len=200)
        prev_name = (
            (raw_prev if is_safe_prev else sanitize_untrusted_str(raw_prev, max_len=200))
            if raw_prev
            else record_path
        )

        raw_patch = all_patches.get(raw_fn, "") if is_safe_path else ""
        additions, deletions = count_patch_line_changes(raw_patch)
        patch_text = raw_patch if len(raw_patch) <= MAX_PATCH_CHARS else ""
        if is_safe_path:
            pr_patches_by_file[raw_fn] = patch_text

        header = f"diff --git a/{prev_name} b/{record_path}\nstatus: {status} (+{additions}/-{deletions})"
        if patch_text:
            diff_chunks.append(f"{header}\n--- a/{prev_name}\n+++ b/{record_path}\n{patch_text}")
        else:
            diff_chunks.append(f"{header}\n(no text patch available)")

        head_file_path_str = None
        base_file_path_str = None
        if idx < 100 and is_safe_path:
            if status != "added":
                base_lookup = raw_prev if (status == "renamed" and raw_prev) else raw_fn
                base_blob = read_blob_by_mode_and_sha(old_mode, old_sha)
                if base_blob is not None:
                    base_file_path_str = safe_write_staged_file(base_root, base_lookup, base_blob)
            if status != "removed":
                head_blob = read_blob_by_mode_and_sha(new_mode, new_sha)
                if head_blob is not None:
                    head_file_path_str = safe_write_staged_file(head_root, raw_fn, head_blob)

        entry_info = {
            "path": record_path,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "head_file_path": head_file_path_str,
            "base_file_path": base_file_path_str,
        }
        if not is_safe_path:
            entry_info["unsafe_path"] = True
        changed_files_info.append(entry_info)

    diff_path.write_text("\n\n".join(diff_chunks) + ("\n" if diff_chunks else ""), encoding="utf-8")
    return (
        str(diff_path.resolve()),
        str(head_root.resolve()),
        str(base_root.resolve()),
        changed_files_info,
        pr_patches_by_file,
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
    if conflicted_files:
        c_diff = run_git(
            ["--literal-pathspecs", "diff", "--no-renames", current_tree, result_tree, "--", *conflicted_files]
        ).stdout
        if c_diff.strip():
            diff_sections.append(c_diff.rstrip())
            per_file_conflicts.update(split_git_diff_by_file(c_diff))
        for fp in conflicted_files:
            if fp not in per_file_conflicts and conflict_notes:
                matching = [n for n in conflict_notes if fp in n] or conflict_notes
                per_file_conflicts[fp] = "\n".join(matching)

    full_conflict_text = ("\n\n".join(diff_sections) + "\n") if diff_sections else ""
    return result_tree, False, conflicted_files, per_file_conflicts, full_conflict_text


def _inspect_mainline_commits(candidates, mainline_ref, base_sha, mainline_dir):
    """Topologically orders `candidates`, inspects patches, and simulates cherry-picking onto `base_sha`."""
    ordered = [(s, by) for s, by in candidates if SHA_RE.fullmatch(s)]
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if is_ancestor(ordered[j][0], ordered[i][0]):
                ordered[i], ordered[j] = ordered[j], ordered[i]

    candidate_sha_set = {s for s, _ in ordered}
    current_tree = base_sha if (SHA_RE.fullmatch(base_sha) and resolve_commit_sha(base_sha)) else None
    commits_info = []
    patches_by_file = {}
    conflicted_set = set()
    conflict_diffs = {}

    for full_sha, matched_by in ordered:
        on_master = is_ancestor(full_sha, mainline_ref)
        parents = get_commit_parents(full_sha)
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

        diff_tree_args = (
            ("diff-tree", "-p", "--no-renames", "-r", parents[0], full_sha)
            if parents
            else ("diff-tree", "-p", "--no-renames", "--no-commit-id", "-r", "--root", full_sha)
        )
        file_patches = git_diff_by_file(*diff_tree_args)

        applied_cleanly = None
        conflicted_files = []
        conflict_diff_file_str = None
        if on_master:
            for fp, patch_body in file_patches.items():
                patches_by_file[fp] = f"{patches_by_file[fp]}\n{patch_body}" if fp in patches_by_file else patch_body
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
                "files_modified": sorted(file_patches.keys()),
                "patch_file": str(patch_file.resolve()),
                "applied_cleanly_on_base": applied_cleanly,
                "conflicted_files": conflicted_files,
                "conflict_diff_file": conflict_diff_file_str,
                "raw_message": raw_message,
            }
        )

    return commits_info, patches_by_file, current_tree, conflicted_set, conflict_diffs


def analyze_mainline_and_compare(
    candidates,
    mainline_ref,
    base_sha,
    head_sha,
    pr_patches_by_file,
    base_root_path,
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
):
    """Inspects mainline commits, simulates cherry-picking onto `base_sha`, and compares Git trees."""
    mainline_dir = stage_dir / "mainline"
    mainline_dir.mkdir(parents=True, exist_ok=True)
    base_root = pathlib.Path(base_root_path)

    (
        mainline_commits_info,
        mainline_patches_by_file,
        current_tree,
        all_conflicted_files,
        conflict_diffs_by_file,
    ) = _inspect_mainline_commits(candidates, mainline_ref, base_sha, mainline_dir)

    pr_diff_entries = get_diff_tree_entries(base_sha, head_sha) or {}
    files_truncated = len(pr_diff_entries) > MAX_AUTO_CLEAN_FILES
    cp_diff_entries = get_diff_tree_entries(base_sha, current_tree) if current_tree is not None else None

    pr_file_set = {p for p in pr_diff_entries if is_safe_repo_rel_path(p)}
    unsafe_paths = sorted(p for p in pr_diff_entries if not is_safe_repo_rel_path(p))
    mainline_file_set = set(mainline_patches_by_file.keys())
    all_files = sorted(pr_file_set | mainline_file_set)

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
            sec.append(f"\n--- Mainline commit patch for {fp} ---")
            sec.append(mainline_patches_by_file.get(fp) or "(empty patch)")

        if fp in conflict_diffs_by_file and not is_lockfile_regen:
            sec.append(f"\n--- 3-way git cherry-pick conflict markers on {base_sha[:10]} for {fp} ---")
            sec.append(conflict_diffs_by_file[fp])

        if in_pr:
            pr_patch = pr_patches_by_file.get(fp) or git_diff_by_file(
                "--literal-pathspecs", "diff", "--no-renames", base_sha, head_sha, "--", fp
            ).get(fp, "")
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

        if not is_lockfile_regen and cp_diff_entries is not None and fp not in all_conflicted_files:
            cp_rec, pr_rec = cp_diff_entries.get(fp), pr_diff_entries.get(fp)
            ref_rec = cp_rec or pr_rec
            if ref_rec is not None:
                base_pair = (ref_rec[0], ref_rec[2])
                cp_pair = (cp_rec[1], cp_rec[3]) if cp_rec else base_pair
                head_pair = (pr_rec[1], pr_rec[3]) if pr_rec else base_pair
                if cp_pair != head_pair:
                    sec.append(
                        f"\n--- Git tree entry mismatch for {fp} ---\n"
                        f"clean cherry-pick: {format_mode_sha(*cp_pair)}\n"
                        f"PR head:           {format_mode_sha(*head_pair)}"
                    )
                cp_blob = read_blob_by_mode_and_sha(*cp_pair)
                head_blob = read_blob_by_mode_and_sha(*head_pair)
                if cp_blob is not None and head_blob is not None and cp_blob != head_blob:
                    deviation = "".join(
                        difflib.unified_diff(
                            cp_blob.splitlines(keepends=True),
                            head_blob.splitlines(keepends=True),
                            fromfile=f"clean_cherry_pick/{fp}",
                            tofile=f"pr_head/{fp}",
                        )
                    )
                    if deviation:
                        sec.append(
                            f"\n--- Exact deviation between clean cherry-pick onto release branch and PR head for {fp} ---"
                        )
                        sec.append(deviation.rstrip())

        return "\n".join(sec)

    for fp in all_files:
        in_pr = fp in pr_file_set
        in_mainline = fp in mainline_file_set
        if in_mainline and not in_pr and base_sha:
            cp_rec = (cp_diff_entries or {}).get(fp)
            base_blob = (
                read_blob_by_mode_and_sha(cp_rec[0], cp_rec[2])
                if cp_rec is not None
                else read_git_regular_file_blob(base_sha, fp)
            )
            if base_blob is not None:
                safe_write_staged_file(base_root, fp, base_blob)

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


def _compute_since_last_review_delta(prev_head_sha, base_ref, base_sha, head_sha, stage_dir):
    """Computes `(pr_patch_unchanged, since_last_review_diff_path)` between `prev_head_sha` and `head_sha`."""
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

    prev_entries = get_diff_tree_entries(prev_base_sha, prev_head_sha) or {}
    curr_entries = get_diff_tree_entries(base_sha, head_sha) or {}
    prev_patches = git_diff_by_file("diff", "--no-renames", prev_base_sha, prev_head_sha)
    curr_patches = git_diff_by_file("diff", "--no-renames", base_sha, head_sha)

    delta_sections = []
    for fp in sorted(set(prev_entries) | set(curr_entries) | set(prev_patches) | set(curr_patches)):
        te_old, te_new = prev_entries.get(fp), curr_entries.get(fp)
        p_old, p_new = prev_patches.get(fp, ""), curr_patches.get(fp, "")
        if prev_base_sha == base_sha and te_old == te_new:
            continue
        has_hunk = "@@" in p_old or "@@" in p_new
        same_across_rebase = (
            te_old is not None
            and te_new is not None
            and (te_old[0], te_old[1], te_old[4]) == (te_new[0], te_new[1], te_new[4])
            and (normalize_patch_changes(p_old) == normalize_patch_changes(p_new) if has_hunk else te_old[3] == te_new[3])
        )
        if same_across_rebase:
            continue

        safe_fp = fp if is_safe_repo_rel_path(fp) else sanitize_untrusted_str(fp, max_len=200)
        patch_delta = "".join(
            difflib.unified_diff(
                p_old.splitlines(keepends=True),
                p_new.splitlines(keepends=True),
                fromfile=f"previous_pr_patch ({prev_head_sha[:10]})/{safe_fp}",
                tofile=f"current_pr_patch ({head_sha[:10]})/{safe_fp}",
            )
        )
        if not patch_delta:
            patch_delta = f"(tree entry changed: {te_old} -> {te_new})"
        delta_sections.append(
            f"=== File patch delta since last review ({prev_head_sha[:10]} -> {head_sha[:10]}): {safe_fp} ===\n"
            f"{patch_delta.rstrip()}"
        )

    if delta_sections:
        since_path.write_text("\n\n".join(delta_sections) + "\n", encoding="utf-8")
    else:
        since_path.write_text(
            f"No effective PR patch changes between last reviewed commit {prev_head_sha} "
            f"and current head {head_sha} (pure rebase or identical diff).\n",
            encoding="utf-8",
        )
    return not delta_sections, str(since_path.resolve())


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
    pr_comments = gh_api_paginated(f"repos/{repo}/issues/{pr_number}/comments?per_page=100", optional=True)
    pr_reviews = gh_api_paginated(f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100", optional=True)
    pr_inline_comments = gh_api_paginated(f"repos/{repo}/pulls/{pr_number}/comments?per_page=100", optional=True)

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

    pr_diff_path, pr_head_root, pr_base_root, changed_files_info, pr_patches_by_file = stage_pr_and_base_files(
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
        pr_patches_by_file=pr_patches_by_file,
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
        pr_patches_by_file=pr_patches_by_file,
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
