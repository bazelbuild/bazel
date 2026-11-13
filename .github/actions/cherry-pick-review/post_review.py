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
import os
import pathlib
import re
import subprocess
import sys

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_INLINE_COMMENTS = 6
ALLOWED_EVENTS = frozenset({"APPROVE", "COMMENT", "REQUEST_CHANGES"})
FILE_LINK_RE = re.compile(r"\[([^\]]+)\]\(file://[^)]+\)")


def gh_api_get(endpoint, paginate=False, slurp=False):
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    if slurp:
        cmd.append("--slurp")
    cmd.append(endpoint)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def gh_api_post(endpoint, payload):
    cmd = ["gh", "api", "--method", "POST", endpoint, "--input", "-"]
    return subprocess.run(
        cmd,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def strip_runner_file_links(markdown_text):
    """Replaces `[label](file:///...)` runner links with `` `label` `` code spans."""
    if not isinstance(markdown_text, str) or "file://" not in markdown_text:
        return markdown_text
    return FILE_LINK_RE.sub(lambda m: f"`{m.group(1).strip('`')}`", markdown_text)


def load_gemini_output():
    """Returns (response_text, envelope_dict) from gemini-artifacts/stdout.log."""
    stdout_log = pathlib.Path("gemini-artifacts/stdout.log")
    if not stdout_log.is_file():
        raise FileNotFoundError("gemini-artifacts/stdout.log not found")
    raw = stdout_log.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return "", {}
    try:
        data = json.loads(raw, strict=False)
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"], data
    except json.JSONDecodeError:
        decoder = json.JSONDecoder(strict=False)
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] == "{":
                try:
                    parsed, _ = decoder.raw_decode(raw[idx:])
                    if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
                        return parsed["response"], parsed
                except json.JSONDecodeError:
                    continue
    return raw, {}


def format_cli_stats(envelope):
    """Formats a single-line summary of Antigravity CLI execution statistics."""
    if not isinstance(envelope, dict) or not envelope:
        return ""
    parts = []
    status = envelope.get("status")
    if status:
        parts.append(f"status={status}")
    duration = envelope.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        parts.append(f"duration={duration:.1f}s")
    num_turns = envelope.get("num_turns")
    if isinstance(num_turns, int) and not isinstance(num_turns, bool):
        parts.append(f"turns={num_turns}")
    usage = envelope.get("usage")
    if isinstance(usage, dict) and usage:
        token_parts = []
        for key, label in (
            ("input_tokens", "in"),
            ("output_tokens", "out"),
            ("thinking_tokens", "thinking"),
            ("cache_read_tokens", "cached"),
            ("total_tokens", "total"),
        ):
            val = usage.get(key)
            if isinstance(val, int) and not isinstance(val, bool):
                token_parts.append(f"{label}={val}")
        if token_parts:
            parts.append(f"tokens({', '.join(token_parts)})")
    denied = envelope.get("denied_actions")
    if isinstance(denied, list) and denied:
        parts.append(f"denied_actions={len(denied)}")
    return ", ".join(parts)


def print_cli_debug_logs(envelope, response_text):
    """Prints CLI stats and collapsible debug groups to stdout for GitHub Actions logs."""
    stats_line = format_cli_stats(envelope)
    if stats_line:
        print(f"Antigravity CLI stats: {stats_line}")

    if response_text:
        print("::group::Gemini model response")
        print(response_text)
        print("::endgroup::")

    for log_name, title in (
        ("gemini-artifacts/stderr.log", "Antigravity CLI stderr.log"),
        ("gemini-artifacts/cli.log", "Antigravity CLI cli.log (tail)"),
    ):
        log_path = pathlib.Path(log_name)
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            if log_text:
                lines = log_text.splitlines()
                tail = "\n".join(lines[-100:])
                print(f"::group::{title}")
                print(tail)
                print("::endgroup::")


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
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return

    lines = ["### 🤖 Gemini Cherry-Pick PR Review", "", f"- **Outcome:** {outcome_line}"]
    if decision:
        lines.append(f"- **Review Decision:** `{decision}`")
    if skip_reason:
        lines.append(f"- **Reason:** {skip_reason}")
    stats_line = format_cli_stats(envelope or {})
    if stats_line:
        lines.append(f"- **CLI Stats:** `{stats_line}`")
    lines.append("")

    if review_body:
        lines.extend(
            [
                "<details open>",
                "<summary><strong>Review Summary</strong></summary>",
                "",
                review_body,
                "",
                "</details>",
                "",
            ]
        )

    if inline_comments:
        lines.extend(
            [
                "<details>",
                f"<summary><strong>Inline Comments ({len(inline_comments)})</strong></summary>",
                "",
                "\n\n".join(format_fallback_comment(c) for c in inline_comments),
                "",
                "</details>",
                "",
            ]
        )

    if response_text:
        fence = "````"
        while fence in response_text:
            fence += "`"
        lines.extend(
            [
                "<details>",
                "<summary><strong>Raw Model Output</strong></summary>",
                "",
                f"{fence}text",
                response_text,
                fence,
                "",
                "</details>",
                "",
            ]
        )

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def dump_gemini_diagnostics(response_text):
    """Logs Gemini stdout/stderr/CLI log diagnostics to stderr when output parsing fails."""
    for log_name in ("gemini-artifacts/stderr.log", "gemini-artifacts/cli.log"):
        log_path = pathlib.Path(log_name)
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            if log_text:
                sys.stderr.write(f"=== {log_name} ===\n{log_text[-4000:]}\n")
    sys.stderr.write(f"=== Gemini response_text (len={len(response_text)}) ===\n{response_text[-4000:]}\n")


def repair_unescaped_json_quotes(s):
    """Escapes unescaped double quotes inside JSON string values."""
    out = []
    in_string = False
    is_key = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_string:
            if ch == "\\":
                if i + 1 < n:
                    out.append(ch)
                    out.append(s[i + 1])
                    i += 2
                    continue
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                rest = s[i + 1 :]
                if is_key:
                    if re.match(r"^\s*:", rest):
                        in_string = False
                        out.append(ch)
                        i += 1
                        continue
                else:
                    if re.match(
                        r'^\s*(?:,\s*"(?:decision|should_post|skip_reason|body|comments|path|line|side|start_line|start_side)"\s*:|\}\s*(?:,\s*\{|\]\s*[,}]|\}\s*(?:```|~~~|$)|$))',
                        rest,
                    ):
                        in_string = False
                        out.append(ch)
                        i += 1
                        continue
                out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
        else:
            if ch == '"':
                prev_non_ws = ""
                for k in range(len(out) - 1, -1, -1):
                    if not out[k].isspace():
                        prev_non_ws = out[k]
                        break
                in_string = True
                is_key = prev_non_ws in ("{", ",")
            out.append(ch)
            i += 1
    return "".join(out)


def extract_review_json(text):
    """Extracts the structured review JSON object from Gemini's response text."""
    text = text.strip()
    if not text:
        dump_gemini_diagnostics(text)
        raise ValueError("Gemini response text is empty")

    decoder = json.JSONDecoder(strict=False)
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] == "{":
            candidate = text[idx:]
            for attempt in (candidate, repair_unescaped_json_quotes(candidate)):
                try:
                    parsed, _ = decoder.raw_decode(attempt)
                    if isinstance(parsed, dict) and ("decision" in parsed or "should_post" in parsed):
                        return parsed
                except json.JSONDecodeError:
                    continue

    dump_gemini_diagnostics(text)
    raise ValueError("Could not find valid review JSON object with 'decision' or 'should_post' in Gemini output")


def normalize_comment_path(raw_path, workspace_root=""):
    """Normalizes a comment file path to a safe repository-relative path."""
    path = str(raw_path or "").strip()
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
    if not path or "\\" in path or "\x00" in path:
        return ""
    parts = path.split("/")
    if any(not p or p in (".", "..", ".git") for p in parts):
        return ""
    return path


def coerce_positive_int(val):
    if isinstance(val, int) and not isinstance(val, bool) and val > 0:
        return val
    if isinstance(val, str) and val.strip().isdigit():
        parsed = int(val.strip())
        if parsed > 0:
            return parsed
    return None


def format_fallback_comment(comment):
    path = normalize_comment_path(comment.get("path")) or "<unknown>"
    line = comment.get("line")
    start_line = comment.get("start_line")
    loc = f"`{path}`"
    if start_line and line and start_line != line:
        loc += f" (lines {start_line}–{line})"
    elif line:
        loc += f" (line {line})"
    body = str(comment.get("body") or "").strip()
    return f"#### {loc}\n\n{body}"


def append_fallback_comments(body, fallback_comments):
    if not fallback_comments:
        return body
    sections = "\n\n".join(format_fallback_comment(c) for c in fallback_comments)
    if body:
        return f"{body.rstrip()}\n\n---\n\n### Inline Review Comments\n\n{sections}"
    return f"### Inline Review Comments\n\n{sections}"


def format_outcome_badge(event, review_url, num_inline=0):
    icon = {"APPROVE": "✅ Approved PR", "REQUEST_CHANGES": "❌ Requested changes on PR"}.get(
        event, "💬 Commented on PR"
    )
    text = f"{icon} ([view review]({review_url}))" if review_url else icon
    if num_inline:
        text += f" with {num_inline} inline comment(s)"
    return text


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

    response_text, envelope = load_gemini_output()
    print_cli_debug_logs(envelope, response_text)
    try:
        review_data = extract_review_json(response_text)
    except ValueError as exc:
        write_step_summary(
            outcome_line=f"❌ Failed to parse review JSON (`{exc}`)",
            envelope=envelope,
            response_text=response_text,
        )
        raise

    should_post = review_data.get("should_post", True) is not False
    skip_reason = str(review_data.get("skip_reason") or "").strip()
    has_existing_review = bool(context.get("has_existing_gemini_review"))
    trigger_type = str(context.get("trigger_type") or "automatic")

    # Guardrail: initial reviews on a PR and manually triggered reviews must always post.
    if not should_post and (not has_existing_review or trigger_type == "manual"):
        print(
            f"Overriding should_post=false -> true (has_existing_gemini_review={has_existing_review}, "
            f"trigger_type={trigger_type})."
        )
        should_post = True

    raw_decision = str(review_data.get("decision") or "COMMENT").strip().upper()
    decision = raw_decision if raw_decision in ALLOWED_EVENTS else "COMMENT"
    body = strip_runner_file_links(str(review_data.get("body") or "").strip())
    raw_comments = review_data.get("comments") or []

    if not should_post:
        print(f"Skipping duplicate review on PR #{pr_number} (skip_reason={skip_reason or 'none'}).")
        write_step_summary(
            outcome_line="⏭️ Skipped posting duplicate review (`should_post: false`)",
            skip_reason=skip_reason or "No substantive changes since the previous Gemini review.",
            review_body=body,
            envelope=envelope,
            response_text=response_text,
        )
        return 0

    changed_files_ctx = context.get("changed_files")
    if isinstance(changed_files_ctx, list):
        pr_filenames = {
            item.get("path")
            for item in changed_files_ctx
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
    else:
        pages = gh_api_get(
            f"repos/{repo}/pulls/{pr_number}/files?per_page=100",
            paginate=True,
            slurp=True,
        )
        pr_filenames = {
            item.get("filename")
            for page in pages
            for item in page
            if isinstance(item, dict) and isinstance(item.get("filename"), str)
        }

    valid_inline_comments = []
    fallback_comments = []
    has_critical_comment = False

    if isinstance(raw_comments, list):
        for item in raw_comments:
            if not isinstance(item, dict):
                continue
            path = normalize_comment_path(item.get("path"), workspace_root=workspace_root)
            c_body = strip_runner_file_links(str(item.get("body") or "").strip())
            line = coerce_positive_int(item.get("line"))
            start_line = coerce_positive_int(item.get("start_line"))
            side = str(item.get("side") or "RIGHT").strip().upper()
            if side not in ("LEFT", "RIGHT"):
                side = "RIGHT"

            if not path or not c_body:
                continue

            if "🔴" in c_body:
                has_critical_comment = True

            valid_line = line is not None
            valid_start = start_line is not None and valid_line and start_line < line

            if path in pr_filenames and valid_line and len(valid_inline_comments) < MAX_INLINE_COMMENTS:
                comment_obj = {
                    "path": path,
                    "line": line,
                    "side": side,
                    "body": c_body,
                }
                if valid_start:
                    comment_obj["start_line"] = start_line
                    start_side = str(item.get("start_side") or side).strip().upper()
                    comment_obj["start_side"] = start_side if start_side in ("LEFT", "RIGHT") else side
                valid_inline_comments.append(comment_obj)
            else:
                fallback_comments.append(
                    {
                        "path": path,
                        "line": line if valid_line else None,
                        "start_line": start_line if valid_start else None,
                        "body": c_body,
                    }
                )

    # Deterministic safety guardrails: enforce REQUEST_CHANGES whenever no mainline commit was found
    # or when a critical finding (🔴 / ❌) was reported in inline comments or the summary body.
    if decision != "REQUEST_CHANGES":
        no_mainline = context.get("overall_cherry_pick_classification") == "NO_MAINLINE_COMMIT"
        body_has_blocker = "🔴" in body or "**Verdict:** ❌" in body
        if no_mainline or has_critical_comment or body_has_blocker:
            if no_mainline:
                reason = "no verified mainline commit on `master` was matched"
            elif has_critical_comment:
                reason = "a critical (`🔴`) inline comment was included"
            else:
                reason = "a critical blocker (`🔴` / `❌`) was reported in the review body"
            print(f"Safety guardrail: overriding {decision} to REQUEST_CHANGES because {reason}.")
            decision = "REQUEST_CHANGES"
            body = re.sub(
                r"\*\*Verdict:\*\*\s*(?:✅\s*Approved|💬\s*Commented[^\n]*)",
                "**Verdict:** ❌ Changes Requested",
                body,
            )
            guardrail_note = (
                f"> ⚠️ **Automated Guardrail:** Review decision set to `REQUEST_CHANGES` because {reason}."
            )
            body = f"{body.rstrip()}\n\n{guardrail_note}" if body else guardrail_note

    final_body = append_fallback_comments(body, fallback_comments)
    if not final_body:
        final_body = f"## 📋 Gemini Cherry-Pick Review Summary\n\n**Decision:** `{decision}`"

    payload = {
        "commit_id": head_sha,
        "event": decision,
        "body": final_body,
        "comments": valid_inline_comments,
    }

    endpoint = f"repos/{repo}/pulls/{pr_number}/reviews"
    res = gh_api_post(endpoint, payload)
    if res.returncode == 0:
        review_resp = json.loads(res.stdout)
        review_url = review_resp.get("html_url")
        print(
            f"Posted Gemini review on PR #{pr_number} "
            f"(event={decision}, inline_comments={len(valid_inline_comments)}, url={review_url})"
        )
        write_step_summary(
            outcome_line=format_outcome_badge(decision, review_url, len(valid_inline_comments)),
            decision=decision,
            review_body=final_body,
            inline_comments=valid_inline_comments,
            envelope=envelope,
            response_text=response_text,
        )
        return 0

    err_output = f"{res.stderr}\n{res.stdout}".strip()
    is_self_review_err = "own pull request" in err_output.lower()

    # Fallback 1: If GitHub rejected inline comment line numbers (HTTP 422, not self-review),
    # fold them into the top-level body.
    if valid_inline_comments and ("422" in err_output or "Unprocessable" in err_output) and not is_self_review_err:
        print(
            f"Warning: GitHub rejected inline comment position(s) ({err_output}); "
            "retrying with inline comments folded into the top-level review body."
        )
        final_body = append_fallback_comments(final_body, valid_inline_comments)
        valid_inline_comments = []
        payload = {
            "commit_id": head_sha,
            "event": decision,
            "body": final_body,
            "comments": [],
        }
        res = gh_api_post(endpoint, payload)
        if res.returncode == 0:
            review_resp = json.loads(res.stdout)
            review_url = review_resp.get("html_url")
            print(f"Posted folded Gemini review on PR #{pr_number} (event={decision}, url={review_url})")
            write_step_summary(
                outcome_line=format_outcome_badge(decision, review_url, 0),
                decision=decision,
                review_body=final_body,
                envelope=envelope,
                response_text=response_text,
            )
            return 0
        err_output = f"{res.stderr}\n{res.stdout}".strip()

    # Fallback 2: If GitHub rejected APPROVE or REQUEST_CHANGES because the reviewer token
    # belongs to the PR author (e.g. @bazel-io reviewing a PR authored by @bazel-io),
    # directly fall back to event="COMMENT".
    if decision in ("APPROVE", "REQUEST_CHANGES") and ("422" in err_output or "Unprocessable" in err_output):
        print(
            f"Warning: GitHub rejected event={decision} ({err_output}); "
            "falling back directly to event=COMMENT."
        )
        comment_body = (
            f"> **Automated Review Decision: `{decision}`** "
            "*(posted as `COMMENT` because the reviewer token matches the PR author)*\n\n"
            f"{final_body}"
        )
        comment_payload = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": comment_body,
            "comments": valid_inline_comments,
        }
        res_comment = gh_api_post(endpoint, comment_payload)
        if res_comment.returncode != 0 and valid_inline_comments:
            folded_comment_body = append_fallback_comments(comment_body, valid_inline_comments)
            comment_payload = {
                "commit_id": head_sha,
                "event": "COMMENT",
                "body": folded_comment_body,
                "comments": [],
            }
            res_comment = gh_api_post(endpoint, comment_payload)
            if res_comment.returncode == 0:
                comment_body = folded_comment_body
                valid_inline_comments = []
        if res_comment.returncode == 0:
            review_resp = json.loads(res_comment.stdout)
            review_url = review_resp.get("html_url")
            print(f"Posted fallback COMMENT review on PR #{pr_number} (url={review_url})")
            write_step_summary(
                outcome_line=format_outcome_badge("COMMENT", review_url, len(valid_inline_comments)),
                decision=f"{decision} (posted as COMMENT)",
                review_body=comment_body,
                inline_comments=valid_inline_comments,
                envelope=envelope,
                response_text=response_text,
            )
            return 0
        res = res_comment

    sys.stderr.write(f"{res.stderr}\n{res.stdout}")
    write_step_summary(
        outcome_line=f"❌ Failed to post PR review (exit code {res.returncode})",
        decision=decision,
        review_body=final_body,
        inline_comments=valid_inline_comments,
        envelope=envelope,
        response_text=response_text,
    )
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
