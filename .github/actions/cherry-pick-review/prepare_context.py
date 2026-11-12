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
import subprocess
import sys

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


def sanitize_untrusted_str(value, max_len=200):
    """Sanitizes untrusted single-line strings before embedding in context.json."""
    text = str(value if value is not None else "")
    text = text.replace("`", "'").replace("@", "@\u200b")
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def sanitize_untrusted_multiline(value, max_len=2000):
    """Sanitizes untrusted multi-line text while preserving line breaks."""
    text = str(value if value is not None else "")
    text = text.replace("```", "'''").replace("@", "@\u200b").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_len:
        return text[:max_len] + "\n..."
    return text


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
    cmd = ["git"] + list(args)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def decode_github_file_content(resp):
    """Decodes base64 file content from GitHub Contents API, rejecting symlinks/dirs."""
    if not isinstance(resp, dict) or resp.get("type") != "file":
        kind = resp.get("type") if isinstance(resp, dict) else type(resp).__name__
        raise ValueError(f"Expected regular file from GitHub API, got {kind!r}")
    if resp.get("encoding") != "base64" or not isinstance(resp.get("content"), str):
        raise ValueError(f"Unsupported or empty content encoding from GitHub API: {resp.get('encoding')!r}")
    return base64.b64decode(resp["content"]).decode("utf-8", errors="replace")


def resolve_path_within(base_dir, rel_path):
    """Returns base_dir / rel_path if it resolves inside base_dir without symlinks, else None."""
    rel_parts = pathlib.PurePosixPath(rel_path).parts
    if not rel_parts or ".." in rel_parts or "." in rel_parts or rel_path.startswith("/"):
        return None
    cur = base_dir
    for part in rel_parts:
        cur = cur / part
        if cur.is_symlink():
            return None
    if not cur.resolve().is_relative_to(base_dir.resolve()):
        return None
    return cur


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
    """Extracts only added/removed lines from a unified diff patch for equivalence checks."""
    if not isinstance(patch_text, str) or not patch_text.strip():
        return []
    changes = []
    for line in patch_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changes.append(line)
    return changes


def split_git_diff_by_file(diff_text):
    """Splits a multi-file `git diff` output into a dict mapping filename -> patch body (from first @@)."""
    result = {}
    if not diff_text:
        return result
    current_file = None
    current_hunks = []
    in_hunks = False
    for line in diff_text.splitlines():
        if line.startswith(("diff --git ", "diff --cc ", "diff --combined ")):
            if current_file is not None:
                result[current_file] = "\n".join(current_hunks)
            m_git = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
            m_cc = re.match(r"^diff --(?:cc|combined) (.+)$", line)
            if m_git:
                current_file = m_git.group(2)
            elif m_cc:
                current_file = m_cc.group(1).strip()
            else:
                current_file = None
            current_hunks = []
            in_hunks = False
        elif current_file is not None:
            if line.startswith("@@"):
                in_hunks = True
            if in_hunks:
                current_hunks.append(line)
    if current_file is not None:
        result[current_file] = "\n".join(current_hunks)
    return result


def detect_mainline_ref():
    """Finds the local git ref representing master/mainline."""
    for candidate in ("origin/master", "master", "origin/main", "main", "HEAD"):
        res = run_git(["rev-parse", "--verify", f"{candidate}^{{commit}}"])
        if res.returncode == 0:
            return candidate
    return "HEAD"


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


def find_mainline_commits(pr_title, pr_body, pr_commits, pr_comments, pr_author, mainline_ref, base_sha):
    """Identifies mainline commits on master referenced by or matching the cherry-pick PR."""
    pr_commit_shas = {str(c.get("sha") or "").strip() for c in pr_commits}
    candidates = []  # list of (full_sha, matched_by)
    seen = set()

    def add_candidate(raw_sha, reason):
        full = resolve_commit_sha(raw_sha)
        if not full or full in seen or full in pr_commit_shas:
            return
        if base_sha and is_ancestor(full, base_sha):
            return
        seen.add(full)
        candidates.append((full, reason))

    commit_messages = [str((c.get("commit") or {}).get("message") or "") for c in pr_commits]
    author_comment_bodies = [
        str(c.get("body") or "")
        for c in pr_comments
        if str((c.get("user") or {}).get("login") or "").lower() == pr_author.lower()
    ]
    all_texts = commit_messages + [pr_body or ""] + author_comment_bodies

    # 1. Explicit `(cherry picked from commit <sha>)` trailers
    for text in all_texts:
        for m in CHERRY_PICK_TRAILER_RE.finditer(text):
            add_candidate(m.group(1), f"cherry-pick trailer ({m.group(1)[:10]})")

    # 2. Explicit `https://github.com/.../commit/<sha>` URLs
    for text in all_texts:
        for m in GITHUB_COMMIT_URL_RE.finditer(text):
            add_candidate(m.group(1), f"GitHub commit URL ({m.group(1)[:10]})")

    # 3. `PiperOrigin-RevId: <id>` trailers
    for text in commit_messages + [pr_body or ""]:
        for m in PIPER_REV_ID_RE.finditer(text):
            rev_id = m.group(1)
            res = run_git(["log", mainline_ref, f"--grep=PiperOrigin-RevId: {rev_id}", "--format=%H", "-n", "5"])
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    add_candidate(line.strip(), f"PiperOrigin-RevId: {rev_id}")

    # 4. `Change-Id: I...` trailers
    if not candidates:
        for text in commit_messages + [pr_body or ""]:
            for m in CHANGE_ID_RE.finditer(text):
                change_id = m.group(1)
                res = run_git(["log", mainline_ref, f"--grep=Change-Id: {change_id}", "--format=%H", "-n", "5"])
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        add_candidate(line.strip(), f"Change-Id: {change_id[:16]}...")

    # 5. Mainline PR number in title `(#12345)`, `COPYBARA_INTEGRATE_REVIEW`, or `Forked from #12345`
    if not candidates:
        pr_nums = []
        m_title = TITLE_PR_RE.search(pr_title or "")
        if m_title:
            pr_nums.append(m_title.group(1))
        for text in commit_messages + [pr_body or ""]:
            for m in COPYBARA_PR_RE.finditer(text):
                pr_nums.append(m.group(1))
            for m in FORKED_FROM_RE.finditer(text):
                pr_nums.append(m.group(1))
        for num in pr_nums:
            for pattern in (f"Closes #{num}", f"(#{num})"):
                res = run_git(["log", mainline_ref, "--fixed-strings", f"--grep={pattern}", "--format=%H", "-n", "5"])
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        add_candidate(line.strip(), f"mainline PR #{num}")

    # 6. Fallback: Commit subject match on master
    if not candidates:
        subjects = []
        cleaned_title = RELEASE_PREFIX_RE.sub("", pr_title or "")
        cleaned_title = TITLE_PR_RE.sub("", cleaned_title).strip()
        if len(cleaned_title) >= 12:
            subjects.append(cleaned_title)
        for msg in commit_messages:
            first_line = msg.splitlines()[0].strip() if msg.strip() else ""
            first_line = RELEASE_PREFIX_RE.sub("", first_line)
            first_line = TITLE_PR_RE.sub("", first_line).strip()
            if len(first_line) >= 12 and first_line.lower() != "update lockfile(s)" and first_line not in subjects:
                subjects.append(first_line)
        for subj in subjects:
            res = run_git(["log", mainline_ref, "--fixed-strings", f"--grep={subj}", "--format=%H", "-n", "5"])
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    subj_check = run_git(["show", "-s", "--format=%s", line.strip()])
                    actual_subj = TITLE_PR_RE.sub("", subj_check.stdout.strip()).strip()
                    if actual_subj == subj or subj in actual_subj:
                        add_candidate(line.strip(), "commit subject match on master")

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
        for m in re.finditer(r"RELNOTES\[INC\]:\s*(.+)", text):
            note = sanitize_untrusted_str(m.group(1), max_len=200)
            if note.lower() not in ("none", "none.", "n/a"):
                add_signal(f"Explicit breaking release note (`RELNOTES[INC]`) in {source_label}: {note}")

        m_api = re.search(
            r"###\s*Build API Changes\s*\n+(.+?)(?:\n###|\nPiperOrigin-RevId:|\nCloses #|\Z)",
            text,
            re.DOTALL,
        )
        if m_api:
            cleaned_api = re.sub(r"<!--.*?-->", "", m_api.group(1), flags=re.DOTALL).strip()
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
                if re.search(r"^[+-]\s*defaultValue\s*=", patch_text, re.MULTILINE):
                    add_signal(
                        f"Modifies `defaultValue` near an incompatible flag definition in `{sanitize_untrusted_str(path, 150)}`."
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
        safe_path = sanitize_untrusted_str(raw_filename, max_len=200)
        status = sanitize_untrusted_str(f.get("status") or "", max_len=40)
        additions = int(f.get("additions") or 0)
        deletions = int(f.get("deletions") or 0)
        patch_text = f.get("patch")
        pr_patches_by_file[raw_filename] = patch_text or ""

        prev_name = sanitize_untrusted_str(f.get("previous_filename") or raw_filename, max_len=200)
        header = f"diff --git a/{prev_name} b/{safe_path}\nstatus: {status} (+{additions}/-{deletions})"
        if isinstance(patch_text, str) and patch_text:
            diff_chunks.append(f"{header}\n--- a/{prev_name}\n+++ b/{safe_path}\n{patch_text}")
        else:
            diff_chunks.append(f"{header}\n(no text patch returned by GitHub API)")

        head_file_path_str = None
        base_file_path_str = None
        if idx < 100:
            # Stage base file from local git base_sha if present
            if status != "added" and base_sha:
                base_dest = resolve_path_within(base_root, raw_filename)
                if base_dest is not None:
                    res_base = run_git(["show", f"{base_sha}:{raw_filename}"])
                    if res_base.returncode == 0:
                        base_dest.parent.mkdir(parents=True, exist_ok=True)
                        base_dest.write_text(res_base.stdout, encoding="utf-8")
                        base_file_path_str = str(base_dest.resolve())

            # Stage head file from local git head_sha, patch reconstruction, or GitHub Contents API
            if status != "removed":
                head_dest = resolve_path_within(head_root, raw_filename)
                if head_dest is not None:
                    content_str = None
                    res_head = run_git(["show", f"{head_sha}:{raw_filename}"])
                    if res_head.returncode == 0:
                        content_str = res_head.stdout
                    if content_str is None and status == "added":
                        content_str = reconstruct_added_file_from_patch(patch_text)
                    if content_str is None:
                        try:
                            resp = gh_api(f"repos/{repo}/contents/{raw_filename}?ref={head_sha}")
                            content_str = decode_github_file_content(resp)
                        except Exception:
                            content_str = None
                    if content_str is not None:
                        head_dest.parent.mkdir(parents=True, exist_ok=True)
                        head_dest.write_text(content_str, encoding="utf-8")
                        head_file_path_str = str(head_dest.resolve())

        changed_files_info.append(
            {
                "path": safe_path,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "head_file_path": head_file_path_str,
                "base_file_path": base_file_path_str,
            }
        )

    diff_path.write_text("\n\n".join(diff_chunks) + ("\n" if diff_chunks else ""), encoding="utf-8")
    return str(diff_path.resolve()), str(head_root.resolve()), str(base_root.resolve()), changed_files_info, pr_patches_by_file


def analyze_mainline_and_compare(
    candidates,
    mainline_ref,
    base_sha,
    pr_files,
    pr_patches_by_file,
    head_root_path,
    stage_dir=pathlib.Path("gemini-artifacts/pr"),
    worktree_dir=pathlib.Path("gemini-artifacts/release_worktree"),
):
    """Inspects mainline commits, simulates cherry-picking onto base_sha, and compares per-file diffs."""
    mainline_dir = stage_dir / "mainline"
    mainline_dir.mkdir(parents=True, exist_ok=True)

    worktree_created = False
    if base_sha and SHA_RE.fullmatch(base_sha):
        run_git(["worktree", "remove", "--force", str(worktree_dir)])
        if worktree_dir.exists():
            import shutil

            shutil.rmtree(worktree_dir, ignore_errors=True)
        res_wt = run_git(["worktree", "add", "--detach", str(worktree_dir), base_sha])
        worktree_created = res_wt.returncode == 0

    mainline_commits_info = []
    mainline_patches_by_file = {}
    all_conflicted_files = set()
    conflict_diffs_by_file = {}
    clean_cherry_pick_files = {}  # path -> content after clean cherry-pick onto base_sha

    for full_sha, matched_by in candidates:
        on_master = is_ancestor(full_sha, mainline_ref)
        subj = run_git(["show", "-s", "--format=%s", full_sha]).stdout.strip()
        author = run_git(["show", "-s", "--format=%an <%ae>", full_sha]).stdout.strip()
        date = run_git(["show", "-s", "--format=%ci", full_sha]).stdout.strip()
        raw_message = run_git(["show", "-s", "--format=%B", full_sha]).stdout.strip()
        show_patch = run_git(["show", "--stat", "-p", full_sha]).stdout

        patch_file = mainline_dir / f"{full_sha}.patch"
        patch_file.write_text(show_patch, encoding="utf-8")

        raw_diff = run_git(["diff-tree", "-p", "--no-commit-id", "-r", "-m", full_sha]).stdout
        file_patches = split_git_diff_by_file(raw_diff)
        for fp, patch_body in file_patches.items():
            if fp in mainline_patches_by_file:
                mainline_patches_by_file[fp] += "\n" + patch_body
            else:
                mainline_patches_by_file[fp] = patch_body

        applied_cleanly = None
        conflicted_files = []
        conflict_diff_file_str = None

        if worktree_created:
            run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
            run_git(["clean", "-fd"], cwd=worktree_dir)
            cp_res = run_git(["cherry-pick", "--no-commit", full_sha], cwd=worktree_dir)
            if cp_res.returncode == 0:
                applied_cleanly = True
                for fp in file_patches:
                    wt_file = resolve_path_within(worktree_dir, fp)
                    if wt_file is not None and wt_file.is_file():
                        clean_cherry_pick_files[fp] = wt_file.read_text(encoding="utf-8", errors="replace")
            else:
                applied_cleanly = False
                unmerged = run_git(["diff", "--name-only", "--diff-filter=U"], cwd=worktree_dir).stdout
                conflicted_files = [line.strip() for line in unmerged.splitlines() if line.strip()]
                all_conflicted_files.update(conflicted_files)
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

        mainline_commits_info.append(
            {
                "sha": full_sha,
                "subject": sanitize_untrusted_str(subj, max_len=200),
                "author": sanitize_untrusted_str(author, max_len=120),
                "date": sanitize_untrusted_str(date, max_len=60),
                "matched_by": matched_by,
                "is_on_master": on_master,
                "files_modified": sorted(file_patches.keys()),
                "patch_file": str(patch_file.resolve()),
                "applied_cleanly_on_base": applied_cleanly,
                "conflicted_files": conflicted_files,
                "conflict_diff_file": conflict_diff_file_str,
                "raw_message": raw_message,
            }
        )

        # If there are multiple mainline commits, also attempt applying all of them in sequence
    if worktree_created and len(candidates) > 1:
        run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
        run_git(["clean", "-fd"], cwd=worktree_dir)
        all_clean = True
        for full_sha, _ in candidates:
            if run_git(["cherry-pick", "--no-commit", full_sha], cwd=worktree_dir).returncode != 0:
                run_git(["cherry-pick", "--abort"], cwd=worktree_dir)
                all_clean = False
                break
        if all_clean:
            for fp in mainline_patches_by_file:
                wt_file = resolve_path_within(worktree_dir, fp)
                if wt_file is not None and wt_file.is_file():
                    clean_cherry_pick_files[fp] = wt_file.read_text(encoding="utf-8", errors="replace")
        run_git(["reset", "--hard", base_sha], cwd=worktree_dir)
        run_git(["clean", "-fd"], cwd=worktree_dir)

    pr_file_set = {str(f.get("filename") or "") for f in pr_files if f.get("filename")}
    mainline_file_set = set(mainline_patches_by_file.keys())
    all_files = sorted(pr_file_set | mainline_file_set)

    file_comparisons = []
    vs_mainline_sections = []
    last_mainline_sha = mainline_commits_info[-1]["sha"] if mainline_commits_info else None

    for fp in all_files:
        safe_fp = sanitize_untrusted_str(fp, max_len=200)
        in_pr = fp in pr_file_set
        in_mainline = fp in mainline_file_set
        pr_patch = pr_patches_by_file.get(fp, "")
        ml_patch = mainline_patches_by_file.get(fp, "")

        if fp in LOCKFILE_PATHS and in_pr:
            classification = "lockfile_regenerated"
        elif in_pr and in_mainline:
            head_file = resolve_path_within(pathlib.Path(head_root_path), fp)
            head_content = (
                head_file.read_text(encoding="utf-8", errors="replace")
                if head_file is not None and head_file.is_file()
                else None
            )
            clean_cp_content = clean_cherry_pick_files.get(fp)
            if clean_cp_content is not None and head_content is not None and clean_cp_content == head_content:
                classification = "identical"
            elif normalize_patch_changes(pr_patch) == normalize_patch_changes(ml_patch):
                classification = "identical"
            elif fp in all_conflicted_files:
                classification = "adapted_with_conflict"
            else:
                classification = "adapted_without_conflict"
        elif in_mainline and not in_pr:
            classification = "omitted_from_pr"
        else:
            classification = "extra_in_pr"

        file_comparisons.append(
            {
                "path": safe_fp,
                "classification": classification,
                "in_pr": in_pr,
                "in_mainline": in_mainline,
                "had_merge_conflict": fp in all_conflicted_files,
            }
        )

        if classification not in ("identical", "lockfile_regenerated"):
            sec = [f"================================================================================"]
            sec.append(f"FILE: {safe_fp}")
            sec.append(f"CLASSIFICATION: {classification}")
            sec.append("================================================================================")

            if base_sha and last_mainline_sha:
                hist_res = run_git(["log", "--oneline", "-n", "15", f"{base_sha}..{last_mainline_sha}", "--", fp])
                if hist_res.returncode == 0 and hist_res.stdout.strip():
                    sec.append(f"\n--- Commits touching {safe_fp} on master ({base_sha[:10]}..{last_mainline_sha[:10]}) ---")
                    sec.append(hist_res.stdout.strip())

            if in_mainline:
                sec.append(f"\n--- Mainline commit patch for {safe_fp} ---")
                sec.append(ml_patch or "(empty patch)")

            if fp in conflict_diffs_by_file:
                sec.append(f"\n--- 3-way git cherry-pick conflict markers on {base_sha[:10]} for {safe_fp} ---")
                sec.append(conflict_diffs_by_file[fp])

            if in_pr:
                sec.append(f"\n--- PR patch for {safe_fp} ---")
                sec.append(pr_patch or "(empty patch)")

            # If a clean cherry-pick of this file succeeded, show exact diff between clean cherry-pick and PR head
            clean_cp_content = clean_cherry_pick_files.get(fp)
            head_file = resolve_path_within(pathlib.Path(head_root_path), fp)
            head_content = (
                head_file.read_text(encoding="utf-8", errors="replace")
                if head_file is not None and head_file.is_file()
                else None
            )
            if clean_cp_content is not None and head_content is not None and clean_cp_content != head_content:
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

    verified_mainline = [mc for mc in mainline_commits_info if mc["is_on_master"]]
    if not verified_mainline:
        overall_classification = "NO_MAINLINE_COMMIT"
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

    worktree_path_str = str(worktree_dir.resolve()) if worktree_created else None
    return (
        mainline_commits_info,
        file_comparisons,
        overall_classification,
        vs_mainline_diff_path,
        worktree_path_str,
    )


def render_review_prompt(
    context_json_str,
    template_path=pathlib.Path(".github/actions/cherry-pick-review/prompt_template.md"),
    styleguide_path=pathlib.Path(".github/actions/cherry-pick-review/styleguide.md"),
    out_path=pathlib.Path(".gemini/prompt.md"),
):
    """Renders .gemini/prompt.md by expanding styleguide and context into prompt_template.md."""
    action_dir = pathlib.Path(__file__).resolve().parent
    if not template_path.is_file():
        fallback_tpl = action_dir / "prompt_template.md"
        if fallback_tpl.is_file():
            template_path = fallback_tpl
    if not styleguide_path.is_file():
        fallback_sg = action_dir / "styleguide.md"
        if fallback_sg.is_file():
            styleguide_path = fallback_sg
    if not template_path.is_file() or not styleguide_path.is_file():
        return False
    prompt_template = template_path.read_text(encoding="utf-8")
    styleguide = styleguide_path.read_text(encoding="utf-8").strip()
    # Expand trusted styleguide.md first, then context.json second so untrusted
    # fields in context.json can never trigger template expansion.
    rendered = prompt_template.replace("@{.github/actions/cherry-pick-review/styleguide.md}", styleguide).replace(
        "@{.gemini/context.json}", context_json_str.strip()
    )
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

    m_rel = RELEASE_BRANCH_RE.fullmatch(base_ref)
    if not m_rel:
        print(f"Skipping PR #{pr_number}: base branch {base_ref!r} is not a release-* branch.")
        write_skip_summary(f"Skipped PR #{pr_number} (base branch `{base_ref}` is not a `release-*` branch)")
        with open(github_output, "a", encoding="utf-8") as out:
            out.write("should_run=false\n")
        return 0

    release_version = m_rel.group(1)

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
            r"^(@bazel-io\s+gemini-review|@gemini-cli\s+/review)\s*",
            "",
            comment_body,
        ).strip()
    additional_context = sanitize_untrusted_str(additional_context, max_len=500)

    pr_author = sanitize_untrusted_str((pr_info.get("user") or {}).get("login", ""), max_len=80)
    pr_title = sanitize_untrusted_str(pr_info.get("title") or "", max_len=250)
    raw_pr_body = str(pr_info.get("body") or "")
    sanitized_pr_body = sanitize_untrusted_multiline(raw_pr_body, max_len=3000)

    # Fetch PR commits, changed files, and comments
    commit_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/commits?per_page=100", paginate=True, slurp=True)
    pr_commits = [item for page in commit_pages for item in page] if isinstance(commit_pages, list) else []

    changed_files_count = int(pr_info.get("changed_files") or 0)
    if changed_files_count > 500:
        pr_files = gh_api(f"repos/{repo}/pulls/{pr_number}/files?per_page=100")
    else:
        file_pages = gh_api(f"repos/{repo}/pulls/{pr_number}/files?per_page=100", paginate=True, slurp=True)
        pr_files = [item for page in file_pages for item in page] if isinstance(file_pages, list) else []

    try:
        comment_pages = gh_api(f"repos/{repo}/issues/{pr_number}/comments?per_page=100", paginate=True, slurp=True)
        pr_comments = [item for page in comment_pages for item in page] if isinstance(comment_pages, list) else []
    except Exception:
        pr_comments = []

    # Ensure master, base_ref, and the PR head are fetched in the local git repository
    run_git(
        [
            "fetch",
            "--filter=blob:none",
            "--no-tags",
            "origin",
            "+refs/heads/master:refs/remotes/origin/master",
            f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
            f"refs/pull/{pr_number}/head",
        ]
    )
    mainline_ref = detect_mainline_ref()

    pr_diff_path, pr_head_root, pr_base_root, changed_files_info, pr_patches_by_file = stage_pr_and_base_files(
        repo=repo,
        base_sha=base_sha,
        head_sha=head_sha,
        pr_files=pr_files,
    )

    candidates = find_mainline_commits(
        pr_title=pr_info.get("title") or "",
        pr_body=raw_pr_body,
        pr_commits=pr_commits,
        pr_comments=pr_comments,
        pr_author=(pr_info.get("user") or {}).get("login", ""),
        mainline_ref=mainline_ref,
        base_sha=base_sha,
    )

    (
        mainline_commits_info,
        file_comparisons,
        overall_classification,
        vs_mainline_diff_path,
        release_worktree_path,
    ) = analyze_mainline_and_compare(
        candidates=candidates,
        mainline_ref=mainline_ref,
        base_sha=base_sha,
        pr_files=pr_files,
        pr_patches_by_file=pr_patches_by_file,
        head_root_path=pr_head_root,
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

    owner, repo_name = repo.split("/", 1)
    context_data = {
        "repository": repo,
        "owner": owner,
        "repo": repo_name,
        "workspace_root": str(pathlib.Path.cwd().resolve()),
        "pull_request_number": int(pr_number),
        "base_ref": base_ref,
        "release_version": release_version,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "pr_title": pr_title,
        "pr_body": sanitized_pr_body,
        "pr_author": pr_author,
        "trigger_type": trigger_type,
        "additional_context": additional_context,
        "pr_commits": serialized_pr_commits,
        "pr_comments": serialized_pr_comments,
        "mainline_commits": serialized_mainline_commits,
        "overall_cherry_pick_classification": overall_classification,
        "file_comparisons": file_comparisons,
        "breaking_change_signals": breaking_change_signals,
        "pr_diff_path": pr_diff_path,
        "vs_mainline_diff_path": vs_mainline_diff_path,
        "pr_head_root": pr_head_root,
        "pr_base_root": pr_base_root,
        "release_worktree_path": release_worktree_path,
        "changed_files": changed_files_info,
    }

    os.makedirs(".gemini", exist_ok=True)
    context_json_str = json.dumps(context_data, indent=2)
    with open(".gemini/context.json", "w", encoding="utf-8") as f:
        f.write(context_json_str)

    render_review_prompt(context_json_str)

    print("Generated .gemini/context.json:")
    print(context_json_str)

    with open(github_output, "a", encoding="utf-8") as out:
        out.write("should_run=true\n")
        out.write(f"pr_number={pr_number}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
