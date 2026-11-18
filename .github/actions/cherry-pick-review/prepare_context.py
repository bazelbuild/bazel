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

import base64
import difflib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PR_NUMBER_RE = re.compile(r"^[1-9]\d*$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHORT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
RELEASE_BRANCH_RE = re.compile(r"^release-([A-Za-z0-9._-]+)$")
LOCKFILE_PATHS = frozenset({"MODULE.bazel.lock", "src/test/tools/bzlmod/MODULE.bazel.lock"})

CHERRY_PICK_TRAILER_RE = re.compile(r"\(cherry picked from commit ([0-9a-fA-F]{7,40})\)")
GITHUB_COMMIT_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/([0-9a-fA-F]{7,40})"
)
PIPER_REV_ID_RE = re.compile(r"PiperOrigin-RevId:\s*(\d+)")
CHANGE_ID_RE = re.compile(r"Change-Id:\s*(I[0-9a-fA-F]{8,64})")
COPYBARA_PR_RE = re.compile(
    r"COPYBARA_INTEGRATE_REVIEW=https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/(\d+)"
)
FORKED_FROM_RE = re.compile(r"Forked from #(\d+)")
TITLE_PR_RE = re.compile(r"\(#(\d+)\)\s*$")
RELEASE_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
GEMINI_REVIEW_MARKER = "Gemini Cherry-Pick Review Summary"
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


def sanitize_untrusted_str(value, max_len=200):
    """Sanitizes untrusted single-line strings before embedding in context.json."""
    text = str(value if value is not None else "")
    text = text.replace("\x00", "").replace("`", "'").replace("@", "@\u200b")
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def sanitize_untrusted_multiline(value, max_len=2000):
    """Sanitizes untrusted multi-line text while preserving line breaks."""
    text = str(value if value is not None else "")
    text = (
        text.replace("\x00", "")
        .replace("```", "'''")
        .replace("@", "@\u200b")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    if len(text) > max_len:
        return text[:max_len] + "\n..."
    return text


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
    if not m:
        return False
    return int(m.group(3)) > 0


def is_safe_repo_rel_path(path):
    """Returns True if path is a safe, normalized repository-relative file path.

    Rejects path traversal (`..`, `.`), `.git` segments, leading/trailing `/`,
    backslashes, control characters, and leading `-` or `:` (which could be
    interpreted by Git as flags or pathspec magic).
    """
    if not isinstance(path, str) or not path or len(path) > 1024:
        return False
    if "\x00" in path or "\r" in path or "\n" in path or "\\" in path:
        return False
    if path.startswith("/") or path.endswith("/") or path.startswith("-") or path.startswith(":"):
        return False
    for part in path.split("/"):
        if not part or part in (".", "..", ".git"):
            return False
    return True


def resolve_path_within(base_dir, rel_path):
    """Returns base_dir / rel_path if it resolves inside base_dir without symlinks, else None."""
    if not is_safe_repo_rel_path(rel_path):
        return None
    if base_dir.is_symlink():
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
        cur = base_dir
        for part in rel_path.split("/")[:-1]:
            cur = cur / part
            if cur.is_symlink() or cur.is_file():
                return None
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
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def run_git(args, cwd=None, check=False):
    cmd = ["git", "-c", "core.quotePath=false"] + list(args)
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def run_git_bytes(args, cwd=None, check=False):
    cmd = ["git", "-c", "core.quotePath=false"] + list(args)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=False, check=check)


def get_git_tree_entry(commit_sha, rel_path, cwd=None):
    """Returns (mode, obj_type, sha) for a file/symlink/gitlink at `commit_sha:rel_path`, or None if absent."""
    if not isinstance(commit_sha, str) or not SHA_RE.fullmatch(commit_sha):
        return None
    if not is_safe_repo_rel_path(rel_path):
        return None
    ls_res = run_git(["--literal-pathspecs", "ls-tree", "-z", commit_sha, "--", rel_path], cwd=cwd)
    if ls_res.returncode != 0 or not ls_res.stdout:
        return None
    entry = ls_res.stdout.split("\0", 1)[0]
    if "\t" not in entry:
        return None
    meta, entry_path = entry.split("\t", 1)
    if entry_path != rel_path:
        return None
    meta_parts = meta.split()
    if len(meta_parts) != 3:
        return None
    mode, obj_type, obj_sha = meta_parts
    if (
        obj_type not in ("blob", "commit")
        or mode not in ("100644", "100755", "120000", "160000")
        or not SHA_RE.fullmatch(obj_sha)
    ):
        return None
    return (mode, obj_type, obj_sha)


def format_tree_entry(entry):
    """Formats a tree entry tuple or '__DELETED__' sentinel for diagnostic output."""
    if entry == "__DELETED__" or entry is None:
        return "deleted"
    if isinstance(entry, tuple) and len(entry) == 3:
        mode, obj_type, obj_sha = entry
        return f"{obj_type} {obj_sha[:12]} (mode {mode})"
    return str(entry)


def read_git_regular_file_blob(commit_sha, rel_path):
    """Reads a non-binary regular file blob (mode 100644/100755) at `commit_sha:rel_path`, rejecting symlinks/binaries."""
    entry = get_git_tree_entry(commit_sha, rel_path)
    if entry is None:
        return None
    mode, obj_type, blob_sha = entry
    if obj_type != "blob" or mode not in ("100644", "100755"):
        return None
    cat_res = run_git_bytes(["cat-file", "blob", blob_sha])
    if cat_res.returncode != 0 or b"\x00" in cat_res.stdout:
        return None
    return cat_res.stdout.decode("utf-8", errors="replace")


def record_worktree_file_state(worktree_dir, rel_path, clean_files, clean_entries):
    """Records the staged Git index entry and decoded text (if regular non-binary file) for `rel_path` in `worktree_dir`."""
    if not is_safe_repo_rel_path(rel_path):
        return
    ls_idx = run_git(["--literal-pathspecs", "ls-files", "--stage", "-z", "--", rel_path], cwd=worktree_dir)
    if ls_idx.returncode != 0:
        return
    items = [item for item in ls_idx.stdout.split("\0") if item]
    if not items:
        clean_entries[rel_path] = "__DELETED__"
        clean_files[rel_path] = "__DELETED__"
        return
    if len(items) != 1 or "\t" not in items[0]:
        return
    meta, idx_path = items[0].split("\t", 1)
    if idx_path != rel_path:
        return
    meta_parts = meta.split()
    if len(meta_parts) != 3:
        return
    mode, obj_sha, stage = meta_parts
    if stage != "0" or not SHA_RE.fullmatch(obj_sha) or mode not in ("100644", "100755", "120000", "160000"):
        return
    obj_type = "commit" if mode == "160000" else "blob"
    clean_entries[rel_path] = (mode, obj_type, obj_sha)
    if mode in ("100644", "100755"):
        cat_res = run_git_bytes(["cat-file", "blob", obj_sha], cwd=worktree_dir)
        if cat_res.returncode == 0 and b"\x00" not in cat_res.stdout:
            clean_files[rel_path] = cat_res.stdout.decode("utf-8", errors="replace")
        else:
            clean_files.pop(rel_path, None)
    else:
        clean_files.pop(rel_path, None)


def decode_github_file_content(resp):
    """Decodes base64 non-binary file content from GitHub Contents API, rejecting symlinks/dirs."""
    if not isinstance(resp, dict) or resp.get("type") != "file":
        kind = resp.get("type") if isinstance(resp, dict) else type(resp).__name__
        raise ValueError(f"Expected regular file from GitHub API, got {kind!r}")
    if resp.get("encoding") != "base64" or not isinstance(resp.get("content"), str):
        raise ValueError(f"Unsupported or empty content encoding from GitHub API: {resp.get('encoding')!r}")
    raw = base64.b64decode(resp["content"])
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def reconstruct_added_file_from_patch(patch_text):
    """Reconstructs a newly added file's content from a single-hunk unified diff patch."""
    if not isinstance(patch_text, str) or not patch_text:
        return None
    lines = patch_text.split("\n")
    if not re.match(r"^@@ -0,0 \+1(?:,\d+)? @@", lines[0]):
        return None
    out_lines = []
    trailing_newline = True
    for line in lines[1:]:
        if line.startswith("@@"):
            return None
        if line.startswith("+"):
            out_lines.append(line[1:])
        elif line.startswith("\\ No newline at end of file"):
            trailing_newline = False
        else:
            return None
    content = "\n".join(out_lines)
    if trailing_newline and out_lines:
        content += "\n"
    return content


def normalize_patch_changes(patch_text):
    """Normalizes a unified diff patch by stripping diff/hunk headers while preserving context, metadata, and +/- lines."""
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
        elif not in_hunk and (
            line.startswith(("+", "-"))
            or line.startswith(
                (
                    "old mode ",
                    "new mode ",
                    "Binary files ",
                )
            )
        ):
            changes.append(line)
    return changes


def split_git_diff_by_file(diff_text):
    """Splits a multi-file `git diff` output into a dict mapping safe filename -> patch body."""
    result = {}
    if not diff_text:
        return result
    current_file = None
    current_hunks = []
    in_hunks = False
    for line in diff_text.splitlines():
        if line.startswith(("diff --git ", "diff --cc ", "diff --combined ", "* Unmerged path ")):
            if current_file is not None and is_safe_repo_rel_path(current_file):
                result[current_file] = "\n".join(current_hunks)
            m_git = re.match(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$', line)
            m_cc = re.match(r"^diff --(?:cc|combined) (.+)$", line)
            m_unmerged = re.match(r"^\* Unmerged path (.+)$", line)
            candidate = None
            if m_git:
                candidate = m_git.group(2).strip('"')
            elif m_cc:
                candidate = m_cc.group(1).strip().strip('"')
            elif m_unmerged:
                candidate = m_unmerged.group(1).strip().strip('"')
            current_file = candidate if is_safe_repo_rel_path(candidate) else None
            current_hunks = [line] if m_unmerged else []
            in_hunks = bool(m_unmerged)
        elif current_file is not None:
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
    if current_file is not None and is_safe_repo_rel_path(current_file):
        result[current_file] = "\n".join(current_hunks)
    return result


def detect_mainline_ref():
    """Finds the local git ref representing master/mainline (never falling back to HEAD)."""
    for candidate in ("origin/master", "master", "origin/main", "main"):
        res = run_git(["rev-parse", "--verify", f"{candidate}^{{commit}}"])
        if res.returncode == 0:
            return candidate
    raise RuntimeError("Could not resolve mainline ref (origin/master or master) in local git repository")


def resolve_commit_sha(sha_prefix):
    if not isinstance(sha_prefix, str) or not SHORT_SHA_RE.fullmatch(sha_prefix.strip()):
        return None
    res = run_git(["rev-parse", "--verify", f"{sha_prefix.strip()}^{{commit}}"])
    if res.returncode == 0:
        full = res.stdout.strip()
        if SHA_RE.fullmatch(full):
            return full
    return None


def is_ancestor(ancestor_sha, descendant_ref):
    res = run_git(["merge-base", "--is-ancestor", ancestor_sha, descendant_ref])
    return res.returncode == 0


def get_commit_parents(commit_sha):
    """Returns list of parent commit SHAs for `commit_sha`."""
    res = run_git(["rev-list", "--parents", "-n", "1", commit_sha])
    if res.returncode != 0:
        return []
    parts = res.stdout.strip().split()
    return parts[1:] if len(parts) > 1 else []


def find_revert_on_master(commit_sha, mainline_ref, excluded_shas=None, _depth=0):
    """Returns the SHA of an active revert or rollback of `commit_sha` on `mainline_ref`, or None."""
    if _depth > 2 or not isinstance(commit_sha, str) or not SHA_RE.fullmatch(commit_sha):
        return None
    short_prefix = commit_sha[:12]
    res = run_git(
        [
            "log",
            f"{commit_sha}..{mainline_ref}",
            "-i",
            "-E",
            f"--grep=((Automated )?rollback of commit {short_prefix}[0-9a-f]*|This reverts commit {short_prefix}[0-9a-f]*)",
            "--format=%H",
            "-n",
            "5",
        ]
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    excluded = excluded_shas or set()
    pattern = re.compile(
        rf"(?:(?:Automated\s+)?rollback\s+of\s+commit\s+{re.escape(short_prefix)}[0-9a-f]*|This\s+reverts\s+commit\s+{re.escape(short_prefix)}[0-9a-f]*)",
        re.IGNORECASE,
    )
    for line in res.stdout.splitlines():
        rev_sha = line.strip()
        if not SHA_RE.fullmatch(rev_sha) or rev_sha in excluded:
            continue
        msg = run_git(["show", "-s", "--format=%B", rev_sha]).stdout
        header_part = msg.split("*** Original change description ***", 1)[0]
        if pattern.search(header_part):
            if find_revert_on_master(rev_sha, mainline_ref, excluded_shas=excluded, _depth=_depth + 1) is None:
                return rev_sha
    return None


def find_mainline_commits(pr_title, pr_body, pr_commits, pr_comments, pr_author, mainline_ref, base_sha):
    """Identifies mainline commits on master referenced by or matching the cherry-pick PR."""
    pr_commit_shas = {str(c.get("sha") or "").strip() for c in pr_commits}
    candidates = []  # list of (full_sha, matched_by)
    seen = set()

    def is_revert_subject(subj_text):
        lower = (subj_text or "").strip().lower()
        return (
            lower.startswith('revert "')
            or lower.startswith("rollback of ")
            or lower.startswith("automated rollback of ")
        )

    def add_candidate(raw_sha, reason, allow_revert=True):
        full = resolve_commit_sha(raw_sha)
        if not full or full in seen or full in pr_commit_shas:
            return False
        if base_sha and is_ancestor(full, base_sha):
            return False
        if not allow_revert:
            subj_res = run_git(["show", "-s", "--format=%s", full])
            if subj_res.returncode == 0 and is_revert_subject(subj_res.stdout):
                return False
        seen.add(full)
        candidates.append((full, reason))
        return True

    def extract_explicit_from_text(text, allow_rev_id=True, allow_commit_url=True):
        found = False
        for m in CHERRY_PICK_TRAILER_RE.finditer(text):
            if add_candidate(m.group(1), f"cherry-pick trailer ({m.group(1)[:10]})"):
                found = True
        if found:
            return True
        if allow_rev_id:
            for m in PIPER_REV_ID_RE.finditer(text):
                rev_id = m.group(1)
                res = run_git(
                    [
                        "log",
                        mainline_ref,
                        "-E",
                        f"--grep=^PiperOrigin-RevId:[[:space:]]*{rev_id}[[:space:]]*$",
                        "--format=%H",
                        "-n",
                        "5",
                    ]
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if add_candidate(line.strip(), f"PiperOrigin-RevId: {rev_id}"):
                            found = True
                            break
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
        for m in CHANGE_ID_RE.finditer(text):
            change_id = m.group(1)
            res = run_git(
                [
                    "log",
                    mainline_ref,
                    "-E",
                    f"--grep=^Change-Id:[[:space:]]*{change_id}[[:space:]]*$",
                    "--format=%H",
                    "-n",
                    "5",
                ]
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if add_candidate(line.strip(), f"Change-Id: {change_id[:16]}...", allow_revert=allow_revert):
                        found = True
                        break
        if found:
            return True

        pr_nums = []
        m_title = TITLE_PR_RE.search(subject_line or "")
        if m_title:
            pr_nums.append(m_title.group(1))
        for m in COPYBARA_PR_RE.finditer(text):
            pr_nums.append(m.group(1))
        for m in FORKED_FROM_RE.finditer(text):
            pr_nums.append(m.group(1))
        for num in pr_nums:
            res = run_git(
                [
                    "log",
                    mainline_ref,
                    "-E",
                    f"--grep=((^|[^0-9])Closes #{num}([^0-9]|$)|\\(#{num}\\))",
                    "--format=%H",
                    "-n",
                    "5",
                ]
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if add_candidate(line.strip(), f"mainline PR #{num}", allow_revert=allow_revert):
                        found = True
                        break
        if found:
            return True

        cleaned_subj = re.sub(r"[\x00-\x1f]+", " ", subject_line or "")
        cleaned_subj = RELEASE_PREFIX_RE.sub("", cleaned_subj)
        cleaned_subj = TITLE_PR_RE.sub("", cleaned_subj).strip()
        if len(cleaned_subj) >= 12 and cleaned_subj.lower() not in GENERIC_COMMIT_SUBJECTS:
            res = run_git(["log", mainline_ref, "--fixed-strings", f"--grep={cleaned_subj}", "--format=%H", "-n", "5"])
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    subj_check = run_git(["show", "-s", "--format=%s", line.strip()])
                    actual_subj = RELEASE_PREFIX_RE.sub("", TITLE_PR_RE.sub("", subj_check.stdout.strip())).strip()
                    if actual_subj == cleaned_subj:
                        if add_candidate(line.strip(), "commit subject match on master", allow_revert=allow_revert):
                            found = True
                            break
        return found

    commit_messages = [str((c.get("commit") or {}).get("message") or "") for c in pr_commits]
    author_comment_bodies = [
        str(c.get("body") or "")
        for c in pr_comments
        if str((c.get("user") or {}).get("login") or "").lower() == pr_author.lower()
    ]

    # 1. Inspect each PR commit individually so multi-commit PRs resolve every commit
    for msg in commit_messages:
        first_line = msg.splitlines()[0].strip() if msg.strip() else ""
        if not extract_explicit_from_text(msg, allow_rev_id=True, allow_commit_url=True):
            extract_fallback_from_text(msg, subject_line=first_line)

    has_commit_matches = len(candidates) > 0

    # 2. Inspect PR body and author comments for explicit references
    body_matched = extract_explicit_from_text(
        pr_body or "", allow_rev_id=True, allow_commit_url=not has_commit_matches
    )
    for comment_text in author_comment_bodies:
        extract_explicit_from_text(
            comment_text, allow_rev_id=False, allow_commit_url=not (has_commit_matches or body_matched)
        )

    # 3. If still no candidates (or PR body/title wasn't matched via commit messages), try PR title/body fallback
    if not candidates and not body_matched:
        extract_fallback_from_text(pr_body or "", subject_line=pr_title or "")

    return candidates


def extract_breaking_change_signals(mainline_commits_info, pr_title, pr_body, pr_files):
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

    for f in pr_files:
        path = str(f.get("filename") or "")
        patch_text = str(f.get("patch") or "")
        if not patch_text:
            continue
        if path.endswith(".java"):
            if "OptionMetadataTag.INCOMPATIBLE_CHANGE" in patch_text or 'name = "incompatible_' in patch_text:
                has_removed_default = bool(re.search(r"^-\s*defaultValue\s*=", patch_text, re.MULTILINE))
                has_added_default = bool(re.search(r"^\+\s*defaultValue\s*=", patch_text, re.MULTILINE))
                has_added_true_default = bool(
                    re.search(r'^\+\s*defaultValue\s*=\s*"true"', patch_text, re.MULTILINE)
                )
                if (has_removed_default and has_added_default) or (
                    not has_removed_default and has_added_true_default
                ):
                    add_signal(
                        f"Flips or sets `defaultValue` on an incompatible flag in `{sanitize_untrusted_str(path, 150)}`."
                    )
            if re.search(r"^-\s*@StarlarkMethod", patch_text, re.MULTILINE) and not re.search(
                r"^\+\s*@StarlarkMethod", patch_text, re.MULTILINE
            ):
                add_signal(
                    f"Removes `@StarlarkMethod` annotation in `{sanitize_untrusted_str(path, 150)}` (potential Starlark API break)."
                )

    return signals


def stage_pr_and_base_files(repo, base_sha, head_sha, pr_files, stage_dir=pathlib.Path("gemini-artifacts/pr")):
    """Stages the PR unified diff, head files, and base files onto local disk for read_file."""
    head_root = stage_dir / "head"
    base_root = stage_dir / "base"
    head_root.mkdir(parents=True, exist_ok=True)
    base_root.mkdir(parents=True, exist_ok=True)
    diff_path = stage_dir / "diff.patch"

    diff_chunks = []
    changed_files_info = []
    pr_patches_by_file = {}

    for idx, f in enumerate(pr_files):
        raw_filename = str(f.get("filename") or "")
        is_safe_path = is_safe_repo_rel_path(raw_filename)
        record_path = raw_filename if is_safe_path else sanitize_untrusted_str(raw_filename, max_len=1024)
        header_path = raw_filename.replace("`", "'") if is_safe_path else record_path
        status = sanitize_untrusted_str(f.get("status") or "", max_len=40)
        additions = int(f.get("additions") or 0)
        deletions = int(f.get("deletions") or 0)
        patch_text = f.get("patch")
        if is_safe_path:
            pr_patches_by_file[raw_filename] = patch_text or ""

        raw_prev_header = str(f.get("previous_filename") or raw_filename)
        prev_name = (
            raw_prev_header.replace("`", "'")
            if is_safe_repo_rel_path(raw_prev_header)
            else sanitize_untrusted_str(raw_prev_header, max_len=1024)
        )
        header = f"diff --git a/{prev_name} b/{header_path}\nstatus: {status} (+{additions}/-{deletions})"
        if isinstance(patch_text, str) and patch_text:
            diff_chunks.append(f"{header}\n--- a/{prev_name}\n+++ b/{header_path}\n{patch_text}")
        else:
            diff_chunks.append(f"{header}\n(no text patch returned by GitHub API)")

        head_file_path_str = None
        base_file_path_str = None
        if idx < 100 and is_safe_path:
            # Stage base file from local git base_sha if present and regular blob
            if status != "added" and base_sha:
                raw_prev = str(f.get("previous_filename") or raw_filename)
                base_lookup_path = (
                    raw_prev if status == "renamed" and is_safe_repo_rel_path(raw_prev) else raw_filename
                )
                base_blob = read_git_regular_file_blob(base_sha, base_lookup_path)
                if base_blob is not None:
                    base_file_path_str = safe_write_staged_file(base_root, raw_filename, base_blob)
                    if base_lookup_path != raw_filename:
                        safe_write_staged_file(base_root, base_lookup_path, base_blob)

            # Stage head file from local git head_sha, falling back to patch reconstruction or GitHub API
            if status != "removed":
                content_str = read_git_regular_file_blob(head_sha, raw_filename)
                if content_str is None and status == "added":
                    maybe_symlink = (
                        additions == 1
                        and isinstance(patch_text, str)
                        and "\\ No newline at end of file" in patch_text
                    )
                    if not maybe_symlink:
                        content_str = reconstruct_added_file_from_patch(patch_text)
                if content_str is None:
                    try:
                        encoded_rel_path = urllib.parse.quote(raw_filename, safe="/")
                        resp = gh_api(f"repos/{repo}/contents/{encoded_rel_path}?ref={head_sha}")
                        content_str = decode_github_file_content(resp)
                    except Exception:
                        content_str = None
                if content_str is not None:
                    head_file_path_str = safe_write_staged_file(head_root, raw_filename, content_str)

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
    return str(diff_path.resolve()), str(head_root.resolve()), str(base_root.resolve()), changed_files_info, pr_patches_by_file


def analyze_mainline_and_compare(
    candidates,
    mainline_ref,
    base_sha,
    pr_files,
    pr_patches_by_file,
    head_root_path,
    base_root_path,
    head_sha="",
    files_truncated=False,
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
):
    """Inspects mainline commits, simulates cherry-picking onto base_sha, and compares per-file diffs."""
    mainline_dir = stage_dir / "mainline"
    mainline_dir.mkdir(parents=True, exist_ok=True)
    base_root = pathlib.Path(base_root_path)

    tmp_wt_parent = tempfile.mkdtemp(prefix="bazel-cp-wt-")
    worktree_dir = pathlib.Path(tmp_wt_parent) / "release_worktree"
    worktree_created = False

    try:
        if base_sha and SHA_RE.fullmatch(base_sha):
            res_wt = run_git(["worktree", "add", "--detach", str(worktree_dir), base_sha])
            worktree_created = res_wt.returncode == 0

        # Sort candidates in ancestor order (oldest -> newest) so mainline_commits_info,
        # mainline_patches_by_file, and sequential cherry-picks always follow chronological order.
        ordered_candidates = [
            (full_sha, matched_by) for full_sha, matched_by in candidates if SHA_RE.fullmatch(full_sha)
        ]
        for i in range(len(ordered_candidates)):
            for j in range(i + 1, len(ordered_candidates)):
                if is_ancestor(ordered_candidates[j][0], ordered_candidates[i][0]):
                    ordered_candidates[i], ordered_candidates[j] = ordered_candidates[j], ordered_candidates[i]

        candidate_sha_set = {full_sha for full_sha, _ in ordered_candidates}
        mainline_commits_info = []
        mainline_patches_by_file = {}
        master_file_commit_counts = {}
        is_merge_by_sha = {}
        all_conflicted_files = set()
        conflict_diffs_by_file = {}
        clean_cherry_pick_files = {}  # path -> decoded text after clean cherry-pick ("__DELETED__" if deleted)
        clean_cherry_pick_entries = {}  # path -> (mode, obj_type, sha) or "__DELETED__"

        for full_sha, matched_by in ordered_candidates:
            on_master = is_ancestor(full_sha, mainline_ref)
            parents = get_commit_parents(full_sha)
            is_merge = len(parents) > 1
            is_merge_by_sha[full_sha] = is_merge
            meta_out = run_git(["show", "-s", "--format=%s%x00%an <%ae>%x00%ci%x00%B", full_sha]).stdout
            meta_parts = meta_out.split("\0", 3)
            if len(meta_parts) == 4:
                subj, author, date, raw_message = (p.strip() for p in meta_parts)
            else:
                subj = run_git(["show", "-s", "--format=%s", full_sha]).stdout.strip()
                author = run_git(["show", "-s", "--format=%an <%ae>", full_sha]).stdout.strip()
                date = run_git(["show", "-s", "--format=%ci", full_sha]).stdout.strip()
                raw_message = run_git(["show", "-s", "--format=%B", full_sha]).stdout.strip()
            show_args = (
                ["show", "-m", "--first-parent", "--stat", "-p", full_sha]
                if is_merge
                else ["show", "--stat", "-p", full_sha]
            )
            show_patch = run_git(show_args).stdout

            patch_file = mainline_dir / f"{full_sha}.patch"
            patch_file.write_text(show_patch, encoding="utf-8")

            diff_tree_args = (
                ["diff-tree", "-p", "--no-renames", "-r", parents[0], full_sha]
                if parents
                else ["diff-tree", "-p", "--no-renames", "--no-commit-id", "-r", "--root", full_sha]
            )
            raw_diff = run_git(diff_tree_args).stdout
            file_patches = split_git_diff_by_file(raw_diff)

            applied_cleanly = None
            conflicted_files = []
            conflict_diff_file_str = None

            # Only include commits verified to be on master in the comparison baseline so unmerged
            # branch commits can never masquerade as clean mainline changes.
            if on_master:
                for fp, patch_body in file_patches.items():
                    master_file_commit_counts[fp] = master_file_commit_counts.get(fp, 0) + 1
                    if fp in mainline_patches_by_file:
                        mainline_patches_by_file[fp] += "\n" + patch_body
                    else:
                        mainline_patches_by_file[fp] = patch_body

                if worktree_created:
                    run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
                    run_git(["clean", "-fd"], cwd=worktree_dir)
                    cp_args = (
                        ["cherry-pick", "--no-commit", "-m", "1", full_sha]
                        if is_merge
                        else ["cherry-pick", "--no-commit", full_sha]
                    )
                    cp_res = run_git(cp_args, cwd=worktree_dir)
                    if cp_res.returncode == 0:
                        applied_cleanly = True
                        for fp in file_patches:
                            record_worktree_file_state(
                                worktree_dir, fp, clean_cherry_pick_files, clean_cherry_pick_entries
                            )
                    else:
                        applied_cleanly = False
                        unmerged = run_git(["diff", "--name-only", "--diff-filter=U"], cwd=worktree_dir).stdout
                        conflicted_files = [
                            line.strip() for line in unmerged.splitlines() if is_safe_repo_rel_path(line.strip())
                        ]
                        conflicted_set = set(conflicted_files)
                        all_conflicted_files.update(conflicted_files)
                        if conflicted_files:
                            for fp in file_patches:
                                if fp not in conflicted_set:
                                    record_worktree_file_state(
                                        worktree_dir, fp, clean_cherry_pick_files, clean_cherry_pick_entries
                                    )
                        conflict_diff = run_git(["diff"], cwd=worktree_dir).stdout
                        if conflict_diff:
                            c_file = mainline_dir / f"{full_sha}_conflicts.diff"
                            c_file.write_text(conflict_diff, encoding="utf-8")
                            conflict_diff_file_str = str(c_file.resolve())
                            for fp, c_patch in split_git_diff_by_file(conflict_diff).items():
                                conflict_diffs_by_file[fp] = c_patch
                        run_git(["cherry-pick", "--abort"], cwd=worktree_dir)

                    run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
                    run_git(["clean", "-fd"], cwd=worktree_dir)

            reverted_by = (
                find_revert_on_master(full_sha, mainline_ref, excluded_shas=candidate_sha_set)
                if on_master
                else None
            )
            mainline_commits_info.append(
                {
                    "sha": full_sha,
                    "subject": sanitize_untrusted_str(subj, max_len=200),
                    "author": sanitize_untrusted_str(author, max_len=120),
                    "date": sanitize_untrusted_str(date, max_len=60),
                    "matched_by": matched_by,
                    "is_on_master": on_master,
                    "reverted_by_on_master": reverted_by,
                    "files_modified": sorted(file_patches.keys()),
                    "patch_file": str(patch_file.resolve()),
                    "applied_cleanly_on_base": applied_cleanly,
                    "conflicted_files": conflicted_files,
                    "conflict_diff_file": conflict_diff_file_str,
                    "raw_message": raw_message,
                }
            )

        # If there are multiple verified master commits, apply them in chronological sequence on base_sha
        ordered_master_shas = [mc["sha"] for mc in mainline_commits_info if mc["is_on_master"]]
        if len(ordered_master_shas) > 1:
            for fp, count in master_file_commit_counts.items():
                if count > 1:
                    clean_cherry_pick_files.pop(fp, None)
                    clean_cherry_pick_entries.pop(fp, None)
        if worktree_created and len(ordered_master_shas) > 1:
            run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
            run_git(["clean", "-fd"], cwd=worktree_dir)
            all_clean = True
            for full_sha in ordered_master_shas:
                cp_seq_args = (
                    ["cherry-pick", "--no-commit", "-m", "1", full_sha]
                    if is_merge_by_sha.get(full_sha)
                    else ["cherry-pick", "--no-commit", full_sha]
                )
                if run_git(cp_seq_args, cwd=worktree_dir).returncode != 0:
                    seq_unmerged = run_git(["diff", "--name-only", "--diff-filter=U"], cwd=worktree_dir).stdout
                    for line in seq_unmerged.splitlines():
                        if is_safe_repo_rel_path(line.strip()):
                            all_conflicted_files.add(line.strip())
                    run_git(["cherry-pick", "--abort"], cwd=worktree_dir)
                    all_clean = False
                    break
            if all_clean:
                all_conflicted_files.clear()
                conflict_diffs_by_file.clear()
                for mc in mainline_commits_info:
                    if mc["is_on_master"]:
                        mc["applied_cleanly_on_base"] = True
                        mc["conflicted_files"] = []
                        mc["conflict_diff_file"] = None
                for fp in mainline_patches_by_file:
                    record_worktree_file_state(
                        worktree_dir, fp, clean_cherry_pick_files, clean_cherry_pick_entries
                    )
            run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
            run_git(["clean", "-fd"], cwd=worktree_dir)
    finally:
        if worktree_created:
            run_git(["worktree", "remove", "--force", str(worktree_dir)])
        shutil.rmtree(tmp_wt_parent, ignore_errors=True)

    unsafe_pr_files = []
    pr_file_set = set()
    pr_file_status = {}
    for f in pr_files:
        raw_fn = str(f.get("filename") or "")
        raw_status = str(f.get("status") or "")
        if not is_safe_repo_rel_path(raw_fn):
            unsafe_pr_files.append(f)
        else:
            pr_file_set.add(raw_fn)
            pr_file_status[raw_fn] = raw_status
        if raw_status == "renamed":
            raw_prev = str(f.get("previous_filename") or "")
            if raw_prev and raw_prev != raw_fn:
                if not is_safe_repo_rel_path(raw_prev):
                    unsafe_pr_files.append({"filename": raw_prev, "status": "removed"})
                else:
                    pr_file_set.add(raw_prev)
                    pr_file_status.setdefault(raw_prev, "removed")
    mainline_file_set = set(mainline_patches_by_file.keys())
    all_files = sorted(pr_file_set | mainline_file_set)

    verified_mainline = [mc for mc in mainline_commits_info if mc["is_on_master"]]
    has_unverified_mainline = any(not mc["is_on_master"] for mc in mainline_commits_info)
    file_comparisons = []
    vs_mainline_sections = []
    last_mainline_sha = verified_mainline[-1]["sha"] if verified_mainline else None
    head_sha_valid = bool(head_sha and SHA_RE.fullmatch(head_sha) and resolve_commit_sha(head_sha))
    base_sha_valid = bool(base_sha and SHA_RE.fullmatch(base_sha) and resolve_commit_sha(base_sha))

    for fp in all_files:
        safe_fp = fp.replace("`", "'")
        in_pr = fp in pr_file_set
        in_mainline = fp in mainline_file_set
        pr_patch = pr_patches_by_file.get(fp, "")
        ml_patch = mainline_patches_by_file.get(fp, "")
        clean_cp_content = clean_cherry_pick_files.get(fp)
        clean_cp_entry = clean_cherry_pick_entries.get(fp)
        head_tree_entry = (get_git_tree_entry(head_sha, fp) or "__DELETED__") if head_sha_valid else None
        head_file = resolve_path_within(pathlib.Path(head_root_path), fp) if in_pr else None
        head_content = (
            head_file.read_text(encoding="utf-8", errors="replace")
            if head_file is not None and head_file.is_file()
            else None
        )

        # Ensure base file is also staged for any mainline file omitted from the PR
        if in_mainline and not in_pr and base_sha:
            base_blob = read_git_regular_file_blob(base_sha, fp)
            if base_blob is not None:
                safe_write_staged_file(base_root, fp, base_blob)

        norm_pr_patch = normalize_patch_changes(pr_patch)
        norm_ml_patch = normalize_patch_changes(ml_patch)

        if in_pr and in_mainline:
            if clean_cp_entry is not None and head_tree_entry is not None:
                if clean_cp_entry == head_tree_entry:
                    classification = "identical"
                elif fp in LOCKFILE_PATHS:
                    classification = "lockfile_regenerated"
                elif fp in all_conflicted_files:
                    classification = "adapted_with_conflict"
                else:
                    classification = "adapted_without_conflict"
            elif clean_cp_content == "__DELETED__":
                if pr_file_status.get(fp) == "removed" and head_content is None:
                    classification = "identical"
                elif fp in LOCKFILE_PATHS:
                    classification = "lockfile_regenerated"
                elif fp in all_conflicted_files:
                    classification = "adapted_with_conflict"
                else:
                    classification = "adapted_without_conflict"
            elif clean_cp_content is not None and head_content is not None:
                if clean_cp_content == head_content:
                    classification = "identical"
                elif fp in LOCKFILE_PATHS:
                    classification = "lockfile_regenerated"
                elif fp in all_conflicted_files:
                    classification = "adapted_with_conflict"
                else:
                    classification = "adapted_without_conflict"
            elif fp not in all_conflicted_files and norm_pr_patch and norm_pr_patch == norm_ml_patch:
                classification = "identical"
            elif fp in LOCKFILE_PATHS:
                classification = "lockfile_regenerated"
            elif fp in all_conflicted_files:
                classification = "adapted_with_conflict"
            else:
                classification = "adapted_without_conflict"
        elif fp in LOCKFILE_PATHS and in_pr:
            classification = "lockfile_regenerated"
        elif in_mainline and not in_pr:
            base_tree_entry = (get_git_tree_entry(base_sha, fp) or "__DELETED__") if base_sha_valid else None
            if (
                clean_cp_entry is not None
                and base_tree_entry is not None
                and clean_cp_entry == base_tree_entry
                and (head_tree_entry is None or head_tree_entry == base_tree_entry)
            ):
                classification = "identical"
            else:
                classification = "omitted_from_pr"
        else:
            classification = "extra_in_pr"

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
            is_lockfile_regen = classification == "lockfile_regenerated"
            sec = ["================================================================================"]
            sec.append(f"FILE: {safe_fp}")
            sec.append(f"CLASSIFICATION: {classification}")
            sec.append("================================================================================")

            if not is_lockfile_regen and base_sha and last_mainline_sha and is_safe_repo_rel_path(fp):
                hist_res = run_git(
                    [
                        "--literal-pathspecs",
                        "log",
                        "--oneline",
                        "-n",
                        "15",
                        f"{base_sha}..{last_mainline_sha}",
                        "--",
                        fp,
                    ]
                )
                if hist_res.returncode == 0 and hist_res.stdout.strip():
                    sec.append(
                        f"\n--- Commits touching {safe_fp} on master ({base_sha[:10]}..{last_mainline_sha[:10]}) ---"
                    )
                    sec.append(hist_res.stdout.strip())

            if in_mainline and not is_lockfile_regen:
                sec.append(f"\n--- Mainline commit patch for {safe_fp} ---")
                sec.append(ml_patch or "(empty patch)")

            if fp in conflict_diffs_by_file and not is_lockfile_regen:
                sec.append(f"\n--- 3-way git cherry-pick conflict markers on {base_sha[:10]} for {safe_fp} ---")
                sec.append(conflict_diffs_by_file[fp])

            if in_pr:
                sec.append(f"\n--- PR patch for {safe_fp} ---")
                if is_lockfile_regen and pr_patch:
                    patch_lines = pr_patch.splitlines()
                    if len(patch_lines) > 200:
                        sec.append("\n".join(patch_lines[:200]) + "\n... (lockfile patch truncated; see pr_diff_path)")
                    else:
                        sec.append(pr_patch)
                else:
                    sec.append(pr_patch or "(empty patch)")

            if (
                not is_lockfile_regen
                and clean_cp_entry is not None
                and head_tree_entry is not None
                and clean_cp_entry != head_tree_entry
            ):
                sec.append(
                    f"\n--- Git tree entry mismatch for {safe_fp} ---\n"
                    f"clean cherry-pick: {format_tree_entry(clean_cp_entry)}\n"
                    f"PR head:           {format_tree_entry(head_tree_entry)}"
                )

            if (
                not is_lockfile_regen
                and clean_cp_content is not None
                and clean_cp_content != "__DELETED__"
                and head_content is not None
                and clean_cp_content != head_content
            ):
                deviation = "".join(
                    difflib.unified_diff(
                        clean_cp_content.splitlines(keepends=True),
                        head_content.splitlines(keepends=True),
                        fromfile=f"clean_cherry_pick/{safe_fp}",
                        tofile=f"pr_head/{safe_fp}",
                    )
                )
                if deviation:
                    sec.append(
                        f"\n--- Exact deviation between clean cherry-pick onto release branch and PR head for {safe_fp} ---"
                    )
                    sec.append(deviation.rstrip())

            vs_mainline_sections.append("\n".join(sec))

    # Surface any PR files with unsafe/non-standard paths (e.g. leading `-` or `:`) without ever
    # passing them to Git or filesystem path operations, so they can never be silently ignored.
    for uf in unsafe_pr_files:
        safe_uf_path = sanitize_untrusted_str(uf.get("filename") or "<invalid-path>", max_len=200)
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
            "================================================================================\n"
            f"FILE: {safe_uf_path}\n"
            "CLASSIFICATION: extra_in_pr (rejected by repository-relative path validator)\n"
            "================================================================================"
        )

    reverted_mainline_pairs = [
        f"{mc['sha'][:12]} (reverted by {mc['reverted_by_on_master'][:12]})"
        for mc in verified_mainline
        if mc.get("reverted_by_on_master")
    ]
    if reverted_mainline_pairs:
        vs_mainline_sections.append(
            "================================================================================\n"
            "NOTE: MAINLINE COMMIT WAS REVERTED OR ROLLED BACK ON MASTER\n"
            f"The following mainline commit(s) were later reverted/rolled back on {mainline_ref}: {', '.join(reverted_mainline_pairs)}\n"
            "================================================================================"
        )
    if has_unverified_mainline:
        unverified_shas = ", ".join(mc["sha"][:12] for mc in mainline_commits_info if not mc["is_on_master"])
        vs_mainline_sections.append(
            "================================================================================\n"
            "NOTE: UNVERIFIED COMMIT(S) REFERENCED BY PR\n"
            f"The following referenced commit(s) are not ancestors of {mainline_ref}: {unverified_shas}\n"
            "================================================================================"
        )
    if files_truncated:
        vs_mainline_sections.append(
            "================================================================================\n"
            "NOTE: PR FILE LIST TRUNCATED BY GITHUB API\n"
            "The pull request touches more files than GitHub's /files endpoint returned; automatic CLEAN classification is disabled.\n"
            "================================================================================"
        )
    if verified_mainline and not pr_files and not vs_mainline_sections:
        vs_mainline_sections.append(
            "================================================================================\n"
            "NOTE: PULL REQUEST HAS NO CHANGED FILES\n"
            "================================================================================"
        )

    if not verified_mainline:
        overall_classification = "NO_MAINLINE_COMMIT"
    elif has_unverified_mainline or files_truncated or unsafe_pr_files or not pr_files:
        overall_classification = "MODIFIED_OR_EXTRA_CHANGES"
    else:
        non_clean = [
            fc for fc in file_comparisons if fc["classification"] not in ("identical", "lockfile_regenerated")
        ]
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
    )


def is_authentic_gemini_review(review_obj, pr_author=""):
    """Returns True if review_obj is a genuine automated Gemini review rather than a quote or author spoof."""
    if not isinstance(review_obj, dict):
        return False
    state = str(review_obj.get("state") or "").upper()
    if state not in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED"):
        return False
    body = str(review_obj.get("body") or "").lstrip()
    has_summary_heading = body.startswith(f"## 📋 {GEMINI_REVIEW_MARKER}") or body.startswith(
        f"## {GEMINI_REVIEW_MARKER}"
    )
    has_self_review_banner = body.startswith("> **Automated Review Decision:")
    has_guardrail_banner = body.startswith("> ⚠️ **Automated Guardrail:**")
    if not (has_summary_heading or has_self_review_banner or has_guardrail_banner):
        return False

    user_info = review_obj.get("user")
    login = str((user_info or {}).get("login") or "").strip().lower() if isinstance(user_info, dict) else ""
    assoc = str(review_obj.get("author_association") or "").strip().upper()
    known_bots = {"bazel-io", "github-actions[bot]", "gemini-code-assist[bot]"}
    if login in known_bots:
        return True
    # Allow maintainer/owner tokens (e.g., personal fork testing with GH_TOKEN) or legacy test fixtures
    # without a user dict, provided a non-bot PR author isn't spoofing a COMMENT review on their own PR.
    if not login:
        return True
    if pr_author and login == pr_author.strip().lower() and not has_self_review_banner and state == "COMMENTED":
        return False
    if assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        return True
    return False


def analyze_prior_review_delta(
    pr_reviews,
    pr_inline_comments,
    base_ref,
    base_sha,
    head_sha,
    pr_patches_by_file,
    pr_author="",
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
):
    """Summarizes prior Gemini reviews/inline comments and computes delta since the last reviewed commit."""
    gemini_reviews = []
    for r in pr_reviews:
        if not is_authentic_gemini_review(r, pr_author=pr_author):
            continue
        state = str(r.get("state") or "").upper()
        body = str(r.get("body") or "")
        gemini_reviews.append(
            {
                "id": r.get("id"),
                "state": sanitize_untrusted_str(state, max_len=40),
                "commit_id": sanitize_untrusted_str(r.get("commit_id") or "", max_len=40),
                "submitted_at": sanitize_untrusted_str(r.get("submitted_at") or "", max_len=40),
                "body": sanitize_untrusted_multiline(body, max_len=2000),
            }
        )

    serialized_inline_comments = []
    for c in pr_inline_comments[-30:]:
        if not isinstance(c, dict):
            continue
        serialized_inline_comments.append(
            {
                "id": c.get("id"),
                "author": sanitize_untrusted_str((c.get("user") or {}).get("login") or "", max_len=80),
                "path": sanitize_untrusted_str(c.get("path") or "", max_len=200),
                "line": c.get("line") or c.get("original_line"),
                "in_reply_to_id": c.get("in_reply_to_id"),
                "created_at": sanitize_untrusted_str(c.get("created_at") or "", max_len=40),
                "body": sanitize_untrusted_multiline(c.get("body") or "", max_len=1000),
            }
        )

    has_existing = len(gemini_reviews) > 0
    last_review = gemini_reviews[-1] if has_existing else None
    pr_patch_unchanged = None
    since_last_review_diff_path = None

    if last_review is not None:
        prev_head_sha = str(last_review.get("commit_id") or "").strip()
        if SHA_RE.fullmatch(prev_head_sha):
            stage_dir.mkdir(parents=True, exist_ok=True)
            since_path = stage_dir / "since_last_review.diff"
            if prev_head_sha == head_sha:
                pr_patch_unchanged = True
                since_path.write_text(
                    f"No changes since last review: commit {head_sha} was already reviewed.\n",
                    encoding="utf-8",
                )
                since_last_review_diff_path = str(since_path.resolve())
            else:
                if not resolve_commit_sha(prev_head_sha):
                    run_git(["fetch", "--filter=blob:none", "--no-tags", "origin", prev_head_sha])
                if resolve_commit_sha(prev_head_sha):
                    base_branch_ref = f"origin/{base_ref}"
                    if run_git(["rev-parse", "--verify", f"{base_branch_ref}^{{commit}}"]).returncode != 0:
                        base_branch_ref = base_sha
                    mb_res = run_git(["merge-base", base_branch_ref, prev_head_sha])
                    prev_base_sha = mb_res.stdout.strip() if mb_res.returncode == 0 else base_sha
                    if SHA_RE.fullmatch(prev_base_sha):
                        prev_diff_raw = run_git(["diff", "--no-renames", prev_base_sha, prev_head_sha]).stdout
                        prev_patches = split_git_diff_by_file(prev_diff_raw)
                        curr_patches = pr_patches_by_file
                        if SHA_RE.fullmatch(base_sha) and resolve_commit_sha(head_sha):
                            curr_diff_res = run_git(["diff", "--no-renames", base_sha, head_sha])
                            if curr_diff_res.returncode == 0:
                                curr_patches = split_git_diff_by_file(curr_diff_res.stdout)
                        all_delta_files = sorted(set(prev_patches.keys()) | set(curr_patches.keys()))
                        delta_sections = []
                        unchanged = True
                        for fp in all_delta_files:
                            p_old = prev_patches.get(fp, "")
                            p_new = curr_patches.get(fp, "")
                            if normalize_patch_changes(p_old) != normalize_patch_changes(p_new):
                                unchanged = False
                                safe_fp = sanitize_untrusted_str(fp, max_len=200)
                                patch_delta = "".join(
                                    difflib.unified_diff(
                                        p_old.splitlines(keepends=True),
                                        p_new.splitlines(keepends=True),
                                        fromfile=f"previous_pr_patch ({prev_head_sha[:10]})/{safe_fp}",
                                        tofile=f"current_pr_patch ({head_sha[:10]})/{safe_fp}",
                                    )
                                )
                                delta_sections.append(
                                    f"=== File patch delta since last review ({prev_head_sha[:10]} -> {head_sha[:10]}): {safe_fp} ===\n"
                                    f"{patch_delta.rstrip()}"
                                )
                        pr_patch_unchanged = unchanged
                        if delta_sections:
                            since_path.write_text("\n\n".join(delta_sections) + "\n", encoding="utf-8")
                        else:
                            since_path.write_text(
                                f"No effective PR patch changes between last reviewed commit {prev_head_sha} "
                                f"and current head {head_sha} (pure rebase or identical diff).\n",
                                encoding="utf-8",
                            )
                        since_last_review_diff_path = str(since_path.resolve())

    return (
        has_existing,
        last_review,
        pr_patch_unchanged,
        since_last_review_diff_path,
        serialized_inline_comments,
    )


def render_review_prompt(
    context_json_str,
    template_path=pathlib.Path(".github/actions/cherry-pick-review/prompt_template.md"),
    out_path=pathlib.Path(".gemini/prompt.md"),
):
    """Renders .gemini/prompt.md by expanding context.json into prompt_template.md."""
    action_dir = pathlib.Path(__file__).resolve().parent
    if not template_path.is_file():
        fallback_tpl = action_dir / "prompt_template.md"
        if fallback_tpl.is_file():
            template_path = fallback_tpl
    if not template_path.is_file():
        return False
    prompt_template = template_path.read_text(encoding="utf-8")
    rendered = prompt_template.replace("@{.gemini/context.json}", context_json_str.strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return True


def write_skip_summary(message):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(f"### 🤖 Gemini Cherry-Pick PR Review\n\n- **Outcome:** ⏭️ {message}\n")


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
        print(f"Skipping PR #{pr_number}: base branch {safe_base_ref!r} is not a valid release-* branch.")
        write_skip_summary(f"Skipped PR #{pr_number} (base branch `{safe_base_ref}` is not a `release-*` branch)")
        with open(github_output, "a", encoding="utf-8") as out:
            out.write("should_run=false\n")
        return 0

    if pr_state != "open" or (is_draft and not is_manual):
        print(f"Skipping PR #{pr_number} (state={pr_state}, draft={is_draft}, trigger={trigger_type})")
        write_skip_summary(
            f"Skipped PR #{pr_number} (`state={pr_state}`, `draft={is_draft}`, `trigger={trigger_type}`)"
        )
        with open(github_output, "a", encoding="utf-8") as out:
            out.write("should_run=false\n")
        return 0

    if not SHA_RE.fullmatch(head_sha):
        raise ValueError(f"Invalid head SHA from GitHub PR API: {head_sha!r}")
    if not SHA_RE.fullmatch(base_sha):
        raise ValueError(f"Invalid base SHA from GitHub PR API: {base_sha!r}")

    additional_context = input_context
    if not additional_context and comment_body:
        additional_context = re.sub(
            r"^@gemini-cli\s+/review\s*",
            "",
            comment_body,
        ).strip()
    additional_context = sanitize_untrusted_str(additional_context, max_len=500)

    pr_author = sanitize_untrusted_str((pr_info.get("user") or {}).get("login", ""), max_len=80)
    pr_author_association = sanitize_untrusted_str(pr_info.get("author_association") or "", max_len=40).upper()
    pr_title = sanitize_untrusted_str(pr_info.get("title") or "", max_len=250)
    raw_pr_body = str(pr_info.get("body") or "")
    sanitized_pr_body = sanitize_untrusted_multiline(raw_pr_body, max_len=3000)

    # Fetch PR commits, changed files, issue comments, reviews, and inline review comments
    commit_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/commits?per_page=100", paginate=True, slurp=True)
    pr_commits = [item for page in commit_pages for item in page] if isinstance(commit_pages, list) else []

    changed_files_count = int(pr_info.get("changed_files") or 0)
    file_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/files?per_page=100", paginate=True, slurp=True)
    pr_files = [item for page in file_pages for item in page] if isinstance(file_pages, list) else []
    files_truncated = changed_files_count > len(pr_files)

    try:
        comment_pages = gh_api(f"repos/{repo}/issues/{pr_number}/comments?per_page=100", paginate=True, slurp=True)
        pr_comments = [item for page in comment_pages for item in page] if isinstance(comment_pages, list) else []
    except Exception:
        pr_comments = []

    try:
        review_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100", paginate=True, slurp=True)
        pr_reviews = [item for page in review_pages for item in page] if isinstance(review_pages, list) else []
    except Exception:
        pr_reviews = []

    try:
        inline_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/comments?per_page=100", paginate=True, slurp=True)
        pr_inline_comments = [item for page in inline_pages for item in page] if isinstance(inline_pages, list) else []
    except Exception:
        pr_inline_comments = []

    # Ensure master, base_ref, and the PR head are fetched when an origin remote is configured,
    # failing closed rather than silently falling back to the checked-out release branch.
    has_origin = run_git(["remote", "get-url", "origin"]).returncode == 0
    if has_origin:
        master_fetch = run_git(
            ["fetch", "--filter=blob:none", "--no-tags", "origin", "+refs/heads/master:refs/remotes/origin/master"]
        )
        if master_fetch.returncode != 0:
            run_git(
                ["fetch", "--filter=blob:none", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"],
                check=True,
            )
        run_git(
            [
                "fetch",
                "--filter=blob:none",
                "--no-tags",
                "origin",
                f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
            ],
            check=True,
        )
        run_git(["fetch", "--filter=blob:none", "--no-tags", "origin", f"refs/pull/{pr_number}/head"], check=True)
    mainline_ref = detect_mainline_ref()
    if not resolve_commit_sha(head_sha):
        raise RuntimeError(f"Could not resolve PR head commit {head_sha} in local git repository")
    mb_res = run_git(["merge-base", base_sha, head_sha])
    effective_base_sha = (
        mb_res.stdout.strip()
        if mb_res.returncode == 0 and SHA_RE.fullmatch(mb_res.stdout.strip())
        else base_sha
    )

    pr_diff_path, pr_head_root, pr_base_root, changed_files_info, pr_patches_by_file = stage_pr_and_base_files(
        repo=repo,
        base_sha=effective_base_sha,
        head_sha=head_sha,
        pr_files=pr_files,
    )

    raw_pr_author = str((pr_info.get("user") or {}).get("login") or "")
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
    ) = analyze_mainline_and_compare(
        candidates=candidates,
        mainline_ref=mainline_ref,
        base_sha=effective_base_sha,
        pr_files=pr_files,
        pr_patches_by_file=pr_patches_by_file,
        head_root_path=pr_head_root,
        base_root_path=pr_base_root,
        head_sha=head_sha,
        files_truncated=files_truncated,
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
        pr_patches_by_file=pr_patches_by_file,
        pr_author=raw_pr_author,
    )

    breaking_change_signals = extract_breaking_change_signals(
        mainline_commits_info=mainline_commits_info,
        pr_title=pr_info.get("title") or "",
        pr_body=raw_pr_body,
        pr_files=pr_files,
    )

    # Strip internal `raw_message` before serializing mainline_commits_info to context.json
    serialized_mainline_commits = []
    for mc in mainline_commits_info:
        mc_copy = dict(mc)
        raw_msg = mc_copy.pop("raw_message", "")
        mc_copy["message_preview"] = sanitize_untrusted_multiline(raw_msg, max_len=1500)
        serialized_mainline_commits.append(mc_copy)

    serialized_pr_commits = [
        {
            "sha": str(c.get("sha") or "")[:40],
            "author": sanitize_untrusted_str(
                ((c.get("author") or {}).get("login"))
                or (((c.get("commit") or {}).get("author") or {}).get("name"))
                or "",
                max_len=80,
            ),
            "message": sanitize_untrusted_multiline((c.get("commit") or {}).get("message") or "", max_len=1000),
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
        int(f.get("additions") or 0) + int(f.get("deletions") or 0)
        for f in pr_files
        if str(f.get("filename") or "") not in LOCKFILE_PATHS
    )
    has_reverted_mainline = any(
        bool(mc.get("reverted_by_on_master")) for mc in mainline_commits_info if mc.get("is_on_master")
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
        "pr_title": pr_title,
        "pr_body": sanitized_pr_body,
        "pr_author": pr_author,
        "pr_author_association": pr_author_association,
        "trigger_type": trigger_type,
        "additional_context": additional_context,
        "files_truncated": files_truncated,
        "non_lockfile_line_changes": non_lockfile_line_changes,
        "has_existing_gemini_review": has_existing_gemini_review,
        "last_gemini_review": last_gemini_review,
        "pr_patch_unchanged_since_last_review": pr_patch_unchanged_since_last_review,
        "since_last_review_diff_path": since_last_review_diff_path,
        "existing_inline_comments": existing_inline_comments,
        "pr_commits": serialized_pr_commits,
        "pr_comments": serialized_pr_comments,
        "mainline_commits": serialized_mainline_commits,
        "has_reverted_mainline": has_reverted_mainline,
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
    with open(".gemini/context.json", "w", encoding="utf-8") as f:
        f.write(context_json_str)

    if not render_review_prompt(context_json_str):
        raise FileNotFoundError("Could not locate prompt_template.md to render .gemini/prompt.md")

    print("Generated .gemini/context.json:")
    print(context_json_str)

    with open(github_output, "a", encoding="utf-8") as out:
        out.write("should_run=true\n")
        out.write(f"pr_number={pr_number}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
