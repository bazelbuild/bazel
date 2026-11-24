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
"""Deterministically validates and posts Gemini's Bazel cherry-pick PR review output."""

import json
import pathlib
import re
import subprocess
import sys

from review_common import (
    AUTOMATED_DECISION_PREFIX,
    GEMINI_REVIEW_MARKER,
    GUARDRAIL_BANNER_PREFIX,
    REPO_RE,
    SELF_REVIEW_REASON_SNIPPET,
    SHA_RE,
    TRUSTED_AUTHOR_ASSOCIATIONS,
    append_step_summary,
    effective_review_state,
    is_safe_repo_rel_path,
    sanitize_untrusted_str,
)

MAX_INLINE_COMMENTS = 6
MAX_EXTERNAL_PATCH_RELEASE_LINES = 300
ALLOWED_EVENTS = frozenset({"APPROVE", "COMMENT", "REQUEST_CHANGES"})
CLI_LOG_FILES = (
    ("gemini-artifacts/stderr.log", "Antigravity CLI stderr.log"),
    ("gemini-artifacts/cli.log", "Antigravity CLI cli.log (tail)"),
)

FILE_LINK_RE = re.compile(r"\[([^\]]+)\]\(file://[^)]+\)")
MARKDOWN_CODE_SPAN_RE = re.compile(
    r"((?:^|\n)[ \t]*`{3,}.*?\n[ \t]*`{3,}|(?:^|\n)[ \t]*~{3,}.*?\n[ \t]*~{3,}|`+[^`]*`+)",
    re.DOTALL,
)
BARE_AT_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_.\-/])@(?=[A-Za-z0-9])")
FENCED_JSON_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*(`{3,}|~{3,})(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n[ \t]*\1[ \t]*(?=\r?\n|$)",
    re.DOTALL,
)
STRUCTURED_BLOCKER_RE = re.compile(
    r"(?:"
    r"\*\*Verdict:\*\*\s*[`*_]*❌"
    r"|(?:^|\n)\s*(?:[-*+]|\d+\.|#+)\s*(?:[`*_]*🔴|\*\*[^*\n]*🔴[^*\n]*\*\*)"
    r"|(?:^|\n)\s*\|[^|\n]+\|\s*[`*_]*❌"
    r")"
)
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def has_structured_blocker_in_body(body):
    """Returns True if body contains a structured ❌ verdict, table ❌, or line-leading 🔴 blocker item."""
    return bool(isinstance(body, str) and body and STRUCTURED_BLOCKER_RE.search(body))


def gh_api_post(endpoint, payload):
    return subprocess.run(
        ["gh", "api", "--method", "POST", endpoint, "--input", "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def strip_runner_file_links(markdown_text):
    """Replaces `[label](file:///...)` runner links with `` `label` `` code spans."""
    if not isinstance(markdown_text, str) or "file://" not in markdown_text:
        return markdown_text
    return FILE_LINK_RE.sub(lambda m: f"`{m.group(1).strip('`')}`", markdown_text)


def neutralize_at_mentions(markdown_text):
    """Inserts a zero-width space after bare @mentions outside fenced or inline code spans."""
    if not isinstance(markdown_text, str) or "@" not in markdown_text:
        return markdown_text
    parts = MARKDOWN_CODE_SPAN_RE.split(markdown_text)
    for idx in range(0, len(parts), 2):
        parts[idx] = BARE_AT_MENTION_RE.sub("@\u200b", parts[idx])
    return "".join(parts)


def load_gemini_output():
    """Returns `(structured_output_or_none, response_text, envelope_dict)` from `gemini-artifacts/stdout.log`."""
    stdout_log = pathlib.Path("gemini-artifacts/stdout.log")
    if not stdout_log.is_file():
        raise FileNotFoundError("gemini-artifacts/stdout.log not found")
    raw = stdout_log.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return None, "", {}
    try:
        envelope = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        envelope = None
        decoder = json.JSONDecoder(strict=False)
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] == "{":
                try:
                    parsed, end_pos = decoder.raw_decode(raw[idx:])
                except json.JSONDecodeError:
                    continue
                if raw[idx + end_pos :].strip() == "" and isinstance(parsed, dict):
                    envelope = parsed
                    break
    if isinstance(envelope, dict) and ("status" in envelope or "response" in envelope or "structured_output" in envelope):
        structured = envelope.get("structured_output")
        response = envelope.get("response") if isinstance(envelope.get("response"), str) else ""
        return (structured if isinstance(structured, dict) else None), response, envelope
    return None, raw, {}


def extract_review_json(text):
    """Fallback parser when `structured_output` is absent (e.g. plain JSON or fenced JSON in `response`)."""
    text = (text or "").strip()
    if not text:
        dump_gemini_diagnostics(text)
        raise ValueError("Gemini output contained neither structured_output nor response text")
    decoder = json.JSONDecoder(strict=False)
    candidates = [m.group(2).strip() for m in reversed(list(FENCED_JSON_BLOCK_RE.finditer(text)))]
    trimmed = re.sub(r"\s*(?:`{3,}|~{3,})\s*$", "", text)
    candidates.extend(trimmed[idx:] for idx, ch in enumerate(trimmed) if ch == "{")
    for cand in candidates:
        if not cand.startswith("{"):
            continue
        try:
            parsed, end_pos = decoder.raw_decode(cand)
        except json.JSONDecodeError:
            continue
        if cand[end_pos:].strip() == "" and isinstance(parsed, dict) and ("decision" in parsed or "should_post" in parsed):
            return parsed
    dump_gemini_diagnostics(text)
    raise ValueError("Could not find valid review JSON object in Gemini output")


def format_cli_stats(envelope):
    """Formats a single-line summary of Antigravity CLI execution statistics."""
    if not isinstance(envelope, dict) or not envelope:
        return ""
    parts = []
    if envelope.get("status"):
        parts.append(f"status={envelope['status']}")
    duration = envelope.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        parts.append(f"duration={duration:.1f}s")
    num_turns = envelope.get("num_turns")
    if isinstance(num_turns, int) and not isinstance(num_turns, bool):
        parts.append(f"turns={num_turns}")
    usage = envelope.get("usage")
    if isinstance(usage, dict):
        token_keys = (
            ("input_tokens", "in"),
            ("output_tokens", "out"),
            ("thinking_tokens", "thinking"),
            ("cache_read_tokens", "cached"),
            ("total_tokens", "total"),
        )
        token_parts = [
            f"{label}={usage[k]}"
            for k, label in token_keys
            if isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
        ]
        if token_parts:
            parts.append(f"tokens({', '.join(token_parts)})")
    denied = envelope.get("denied_actions")
    if isinstance(denied, list) and denied:
        parts.append(f"denied_actions={len(denied)}")
    return ", ".join(parts)


def _read_log_file(log_name):
    log_path = pathlib.Path(log_name)
    return log_path.read_text(encoding="utf-8", errors="replace").strip() if log_path.is_file() else ""


def print_cli_debug_logs(envelope, display_output):
    """Prints CLI stats and collapsible debug groups to stdout for GitHub Actions logs."""
    stats_line = format_cli_stats(envelope)
    if stats_line:
        print(f"Antigravity CLI stats: {stats_line}")
    if display_output:
        print(f"::group::Gemini model response\n{display_output}\n::endgroup::")
    for log_name, title in CLI_LOG_FILES:
        log_text = _read_log_file(log_name)
        if log_text:
            tail = "\n".join(log_text.splitlines()[-100:])
            print(f"::group::{title}\n{tail}\n::endgroup::")


def dump_gemini_diagnostics(response_text):
    """Logs CLI stderr/log tails and response tail when review JSON parsing fails."""
    for log_name, _ in CLI_LOG_FILES:
        log_text = _read_log_file(log_name)
        if log_text:
            sys.stderr.write(f"=== {log_name} ===\n{log_text[-4000:]}\n")
    sys.stderr.write(f"=== Gemini response_text (len={len(response_text)}) ===\n{response_text[-4000:]}\n")


def _details_section(title, body_text, *, open_by_default=False):
    tag = "<details open>" if open_by_default else "<details>"
    return [tag, f"<summary><strong>{title}</strong></summary>", "", body_text, "", "</details>", ""]


def write_step_summary(
    *,
    outcome_line,
    decision="",
    skip_reason="",
    review_body="",
    inline_comments=None,
    envelope=None,
    response_text="",
):
    """Appends a Markdown summary of the Gemini review run to GITHUB_STEP_SUMMARY."""
    lines = [f"- **Outcome:** {outcome_line}"]
    if decision:
        lines.append(f"- **Review Decision:** `{decision}`")
    if skip_reason:
        lines.append(f"- **Reason:** {skip_reason}")
    stats_line = format_cli_stats(envelope or {})
    if stats_line:
        lines.append(f"- **CLI Stats:** `{stats_line}`")
    lines.append("")

    if review_body:
        lines.extend(_details_section("Review Summary", review_body, open_by_default=True))
    if inline_comments:
        lines.extend(
            _details_section(
                f"Inline Comments ({len(inline_comments)})",
                "\n\n".join(format_fallback_comment(c) for c in inline_comments),
            )
        )
    if response_text:
        fence = "````"
        while fence in response_text:
            fence += "`"
        lines.extend(_details_section("Raw Model Output", f"{fence}text\n{response_text}\n{fence}"))

    append_step_summary(lines)


def normalize_comment_path(raw_path, workspace_root=""):
    """Normalizes a comment file path to a safe repository-relative path."""
    path = str(raw_path or "").replace("\u200b", "").strip()
    if path.startswith("file://"):
        path = path[len("file://") :]
    path = path.split("#", 1)[0]
    for marker in ("gemini-artifacts/pr/head/", "gemini-artifacts/pr/base/"):
        idx = path.find(marker)
        if idx != -1:
            path = path[idx + len(marker) :]
    if workspace_root:
        ws_prefix = workspace_root.rstrip("/") + "/"
        if path.startswith(ws_prefix):
            path = path[len(ws_prefix) :]
    path = path.lstrip("/")
    return path if is_safe_repo_rel_path(path) else ""


def parse_diff_hunk_lines(diff_path_str):
    """Parses `pr_diff_path` into `{path: {(side, line_no): hunk_id}}`, or None if unavailable."""
    if not diff_path_str:
        return None
    diff_path = pathlib.Path(diff_path_str)
    if not diff_path.is_file():
        return None
    result = {}
    current_file = None
    minus_a_file = None
    in_hunk = False
    hunk_id = old_ln = new_ln = 0
    for line in diff_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            current_file = minus_a_file = None
            in_hunk = False
            continue
        if not in_hunk:
            if line.startswith("--- a/"):
                minus_a_file = line[len("--- a/") :]
                continue
            if line.startswith("+++ b/"):
                candidate = line[len("+++ b/") :]
                if is_safe_repo_rel_path(candidate):
                    current_file = candidate
                    result.setdefault(current_file, {})
                continue
            if line == "+++ /dev/null" and minus_a_file and is_safe_repo_rel_path(minus_a_file):
                current_file = minus_a_file
                result.setdefault(current_file, {})
                continue
        m_hunk = HUNK_HEADER_RE.match(line)
        if m_hunk and current_file is not None:
            hunk_id += 1
            old_ln, new_ln = int(m_hunk.group(1)), int(m_hunk.group(2))
            in_hunk = True
            continue
        if in_hunk and current_file is not None:
            f_map = result[current_file]
            if line.startswith("+"):
                f_map[("RIGHT", new_ln)] = hunk_id
                new_ln += 1
            elif line.startswith("-"):
                f_map[("LEFT", old_ln)] = hunk_id
                old_ln += 1
            elif line.startswith(" "):
                f_map[("LEFT", old_ln)] = f_map[("RIGHT", new_ln)] = hunk_id
                old_ln += 1
                new_ln += 1
            elif not line.startswith("\\"):
                in_hunk = False
    return result


def coerce_positive_int(val):
    if isinstance(val, int) and not isinstance(val, bool) and val > 0:
        return val
    if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip()) or None
    return None


def format_fallback_comment(comment):
    path = comment.get("display_path") or normalize_comment_path(comment.get("path")) or "General"
    line, start_line = comment.get("line"), comment.get("start_line")
    loc = f"`{path}`"
    if start_line and line and start_line != line:
        loc += f" (lines {start_line}–{line})"
    elif line:
        loc += f" (line {line})"
    return f"#### {loc}\n\n{str(comment.get('body') or '').strip()}"


def append_fallback_comments(body, fallback_comments):
    if not fallback_comments:
        return body
    sections = "### Inline Review Comments\n\n" + "\n\n".join(format_fallback_comment(c) for c in fallback_comments)
    return f"{body.rstrip()}\n\n---\n\n{sections}" if body else sections


def format_outcome_badge(event, review_url, num_inline=0):
    icon = {"APPROVE": "✅ Approved PR", "REQUEST_CHANGES": "❌ Requested changes on PR"}.get(
        event, "💬 Commented on PR"
    )
    text = f"{icon} ([view review]({review_url}))" if review_url else icon
    return f"{text} with {num_inline} inline comment(s)" if num_inline else text


def validate_review_comments(raw_comments, pr_filenames, unsafe_pr_paths, diff_hunk_lines, workspace_root):
    """Splits raw model comments into `(valid_inline_comments, fallback_comments, has_critical_comment)`."""
    valid_inline = []
    fallbacks = []
    has_critical = False
    if not isinstance(raw_comments, list):
        return valid_inline, fallbacks, has_critical

    for item in raw_comments:
        if not isinstance(item, dict):
            continue
        c_body = neutralize_at_mentions(strip_runner_file_links(str(item.get("body") or "").strip()))
        if not c_body:
            continue
        if c_body.lstrip(" \t`*_").startswith("🔴"):
            has_critical = True

        raw_path_str = str(item.get("path") or "").strip()
        path = normalize_comment_path(raw_path_str, workspace_root=workspace_root)
        line = coerce_positive_int(item.get("line"))
        start_line = coerce_positive_int(item.get("start_line"))
        side = str(item.get("side") or "RIGHT").strip().upper()
        side = side if side in ("LEFT", "RIGHT") else "RIGHT"
        start_side = str(item.get("start_side") or side).strip().upper()
        start_side = start_side if start_side in ("LEFT", "RIGHT") else side

        if raw_path_str and not path:
            sanitized_unsafe = sanitize_untrusted_str(raw_path_str, max_len=200)
            if sanitized_unsafe in unsafe_pr_paths:
                fallbacks.append({"display_path": sanitized_unsafe, "line": line, "start_line": None, "body": c_body})
            continue

        valid_line = line is not None
        valid_start = start_line is not None and valid_line and start_line < line
        hunk_ok = True
        if valid_line and diff_hunk_lines is not None:
            f_hunks = diff_hunk_lines.get(path)
            end_hunk = f_hunks.get((side, line)) if f_hunks is not None else None
            if end_hunk is None:
                hunk_ok = False
            elif valid_start and f_hunks.get((start_side, start_line)) != end_hunk:
                valid_start = False

        if path and path in pr_filenames and valid_line and hunk_ok and len(valid_inline) < MAX_INLINE_COMMENTS:
            comment_obj = {"path": path, "line": line, "side": side, "body": c_body}
            if valid_start:
                comment_obj["start_line"] = start_line
                comment_obj["start_side"] = start_side
            valid_inline.append(comment_obj)
        else:
            fallbacks.append(
                {
                    "path": path,
                    "line": line if valid_line else None,
                    "start_line": start_line if valid_start else None,
                    "body": c_body,
                }
            )

    return valid_inline, fallbacks, has_critical


def submit_review_with_fallbacks(endpoint, head_sha, decision, body, inline_comments):
    """Posts the review via `gh api`, falling back on HTTP 422 for out-of-diff positions or self-reviews."""
    def post(event, review_body, comments):
        return gh_api_post(endpoint, {"commit_id": head_sha, "event": event, "body": review_body, "comments": comments})

    def err_text(res):
        return f"{res.stderr}\n{res.stdout}".strip()

    def is_422(res):
        text = err_text(res)
        return "422" in text or "Unprocessable" in text

    cur_event, cur_body, cur_comments = decision, body, list(inline_comments)
    res = post(cur_event, cur_body, cur_comments)

    if res.returncode != 0 and cur_comments and is_422(res) and "own pull request" not in err_text(res).lower():
        print(
            f"Warning: GitHub rejected inline comment position(s) ({err_text(res)}); "
            "retrying with inline comments folded into the top-level review body."
        )
        cur_body = append_fallback_comments(cur_body, cur_comments)
        cur_comments = []
        res = post(cur_event, cur_body, cur_comments)

    if res.returncode != 0 and cur_event in ("APPROVE", "REQUEST_CHANGES") and is_422(res):
        print(f"Warning: GitHub rejected event={cur_event} ({err_text(res)}); falling back directly to event=COMMENT.")
        cur_body = (
            f"{AUTOMATED_DECISION_PREFIX} `{decision}`** "
            f"*(posted as `COMMENT` because the {SELF_REVIEW_REASON_SNIPPET})*\n\n"
            f"{cur_body}"
        )
        cur_event = "COMMENT"
        res = post(cur_event, cur_body, cur_comments)
        if res.returncode != 0 and cur_comments:
            folded = append_fallback_comments(cur_body, cur_comments)
            res_folded = post(cur_event, folded, [])
            if res_folded.returncode == 0:
                return res_folded, cur_event, folded, []

    return res, cur_event, cur_body, cur_comments


def main():
    context_path = pathlib.Path(".gemini/context.json")
    if not context_path.is_file():
        raise FileNotFoundError(".gemini/context.json not found")

    context = json.loads(context_path.read_text(encoding="utf-8"))
    repo = str(context.get("repository") or "").strip()
    pr_number = int(context["pull_request_number"])
    head_sha = str(context.get("head_sha") or "").strip()
    workspace_root = str(context.get("workspace_root") or "").strip()

    if not REPO_RE.fullmatch(repo):
        raise ValueError(f"Invalid repository in context.json: {repo!r}")
    if pr_number <= 0:
        raise ValueError(f"Invalid pull_request_number in context.json: {pr_number!r}")
    if not SHA_RE.fullmatch(head_sha):
        raise ValueError(f"Invalid head_sha in context.json: {head_sha!r}")

    structured_output, response_text, envelope = load_gemini_output()
    raw_display = json.dumps(structured_output, indent=2) if structured_output is not None else response_text
    print_cli_debug_logs(envelope, raw_display)
    try:
        review_data = structured_output if structured_output is not None else extract_review_json(response_text)
    except ValueError as exc:
        write_step_summary(
            outcome_line=f"❌ Failed to parse review JSON (`{exc}`)",
            envelope=envelope,
            response_text=raw_display,
        )
        raise

    should_post = review_data.get("should_post", True) is not False
    skip_reason = str(review_data.get("skip_reason") or "").strip()
    has_existing_review = bool(context.get("has_existing_gemini_review"))
    trigger_type = str(context.get("trigger_type") or "automatic")
    last_review = context.get("last_gemini_review") if isinstance(context.get("last_gemini_review"), dict) else {}
    last_state = effective_review_state(last_review)
    patch_unchanged = context.get("pr_patch_unchanged_since_last_review") is True

    raw_decision = str(review_data.get("decision") or "COMMENT").strip().upper()
    decision = raw_decision if raw_decision in ALLOWED_EVENTS else "COMMENT"
    body = neutralize_at_mentions(strip_runner_file_links(str(review_data.get("body") or "").strip()))

    changed_files_ctx = context.get("changed_files") or []
    pr_filenames = {
        item["path"]
        for item in changed_files_ctx
        if isinstance(item, dict) and isinstance(item.get("path"), str) and not item.get("unsafe_path")
    }
    unsafe_pr_paths = {
        item["path"]
        for item in [*changed_files_ctx, *(context.get("file_comparisons") or [])]
        if isinstance(item, dict) and item.get("unsafe_path") and isinstance(item.get("path"), str)
    }
    diff_hunk_lines = parse_diff_hunk_lines(str(context.get("pr_diff_path") or "").strip())
    valid_inline_comments, fallback_comments, has_critical_comment = validate_review_comments(
        review_data.get("comments") or [],
        pr_filenames=pr_filenames,
        unsafe_pr_paths=unsafe_pr_paths,
        diff_hunk_lines=diff_hunk_lines,
        workspace_root=workspace_root,
    )

    raw_author = str(context.get("pr_author") or "").replace("\u200b", "").strip().lower()
    author_assoc = str(context.get("pr_author_association") or "").strip().upper()
    is_trusted_author = raw_author == "bazel-io" or author_assoc in TRUSTED_AUTHOR_ASSOCIATIONS

    classification = context.get("overall_cherry_pick_classification")
    no_mainline = classification == "NO_MAINLINE_COMMIT"
    force_no_mainline_block = no_mainline and not is_trusted_author
    body_has_blocker = has_structured_blocker_in_body(body)
    would_request_changes = (
        decision == "REQUEST_CHANGES" or force_no_mainline_block or has_critical_comment or body_has_blocker
    )

    oversized_patch_backport = (
        bool(context.get("is_patch_release"))
        and int(context.get("non_lockfile_line_changes") or 0) > MAX_EXTERNAL_PATCH_RELEASE_LINES
    )
    is_clean_unflagged = (
        classification == "CLEAN"
        and not bool(context.get("files_truncated"))
        and not bool(context.get("has_lockfile_changes"))
        and not bool(context.get("breaking_change_signals"))
        and not bool(context.get("has_reverted_mainline"))
        and not oversized_patch_backport
    )
    can_approve = not no_mainline and (is_clean_unflagged or is_trusted_author)

    if not should_post:
        had_active_approval = last_state == "APPROVED" or bool(context.get("has_prior_active_approval"))
        # Covers external PRs where an earlier APPROVE was capped to COMMENT with an Automated Review Decision banner.
        had_approve_banner = str(last_review.get("body") or "").lstrip().startswith(
            f"{AUTOMATED_DECISION_PREFIX} `APPROVE`**"
        )
        stale_approval = not patch_unchanged and (
            (had_active_approval and (decision != "APPROVE" or would_request_changes or not can_approve))
            or (had_approve_banner and (decision != "APPROVE" or would_request_changes))
        )
        unposted_blocker = would_request_changes and last_state != "CHANGES_REQUESTED"
        if not has_existing_review or trigger_type == "manual" or stale_approval or unposted_blocker:
            print(
                f"Overriding should_post=false -> true (has_existing_gemini_review={has_existing_review}, "
                f"trigger_type={trigger_type}, last_state={last_state!r}, "
                f"patch_unchanged={patch_unchanged}, would_request_changes={would_request_changes}, "
                f"can_approve={can_approve})."
            )
            should_post = True

    if not should_post:
        print(f"Skipping duplicate review on PR #{pr_number} (skip_reason={skip_reason or 'none'}).")
        write_step_summary(
            outcome_line="⏭️ Skipped posting duplicate review (`should_post: false`)",
            skip_reason=skip_reason or "No substantive changes since the previous Gemini review.",
            review_body=body,
            envelope=envelope,
            response_text=raw_display,
        )
        return 0

    if body and not body.startswith((f"## 📋 {GEMINI_REVIEW_MARKER}", f"## {GEMINI_REVIEW_MARKER}")):
        body = f"## 📋 {GEMINI_REVIEW_MARKER}\n\n{body}"

    if decision != "REQUEST_CHANGES" and (force_no_mainline_block or has_critical_comment or body_has_blocker):
        if force_no_mainline_block:
            reason = "no verified mainline commit on `master` was matched"
        elif has_critical_comment:
            reason = "a critical (`🔴`) inline comment was included"
        else:
            reason = "a critical blocker (`🔴` / `❌`) was reported in the review body"
        print(f"Safety guardrail: overriding {decision} to REQUEST_CHANGES because {reason}.")
        decision = "REQUEST_CHANGES"
        body = re.sub(r"\*\*Verdict:\*\*\s*[`*_]*(?:✅|💬|⚠️)[^\n]*", "**Verdict:** ❌ Changes Requested", body)
        guardrail_note = f"{GUARDRAIL_BANNER_PREFIX} Review decision set to `REQUEST_CHANGES` because {reason}."
        body = f"{body.rstrip()}\n\n{guardrail_note}" if body else guardrail_note

    if decision == "APPROVE" and not can_approve:
        print(
            f"Privilege guardrail: capping APPROVE to COMMENT "
            f"(classification={classification!r}, no_mainline={no_mainline}, author_association={author_assoc!r})."
        )
        decision = "COMMENT"
        detail = (
            "because no matching mainline commit on `master` was found; please verify this release-branch change before merging"
            if no_mainline
            else "so a maintainer can verify any adaptations, lockfile updates, scope, or API changes before merging"
        )
        note = f"{AUTOMATED_DECISION_PREFIX} `APPROVE`** *(posted as `COMMENT` {detail})*"
        body = f"{note}\n\n{body}" if body else note

    final_body = append_fallback_comments(body, fallback_comments)
    if not final_body:
        final_body = f"## 📋 {GEMINI_REVIEW_MARKER}\n\n**Decision:** `{decision}`"

    endpoint = f"repos/{repo}/pulls/{pr_number}/reviews"
    res, posted_event, posted_body, posted_comments = submit_review_with_fallbacks(
        endpoint, head_sha, decision, final_body, valid_inline_comments
    )
    if res.returncode == 0:
        review_url = json.loads(res.stdout).get("html_url")
        decision_label = decision if posted_event == decision else f"{decision} (posted as {posted_event})"
        print(
            f"Posted Gemini review on PR #{pr_number} "
            f"(event={posted_event}, inline_comments={len(posted_comments)}, url={review_url})"
        )
        write_step_summary(
            outcome_line=format_outcome_badge(posted_event, review_url, len(posted_comments)),
            decision=decision_label,
            review_body=posted_body,
            inline_comments=posted_comments,
            envelope=envelope,
            response_text=raw_display,
        )
        return 0

    sys.stderr.write(f"{res.stderr}\n{res.stdout}")
    write_step_summary(
        outcome_line=f"❌ Failed to post PR review (exit code {res.returncode})",
        decision=decision,
        review_body=final_body,
        inline_comments=valid_inline_comments,
        envelope=envelope,
        response_text=raw_display,
    )
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
