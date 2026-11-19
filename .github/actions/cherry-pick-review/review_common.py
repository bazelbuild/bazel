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
"""Shared constants and path/string sanitizers for Bazel cherry-pick PR review scripts."""

import re

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PR_NUMBER_RE = re.compile(r"^[1-9]\d*$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHORT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
LOCKFILE_PATHS = frozenset({"MODULE.bazel.lock", "src/test/tools/bzlmod/MODULE.bazel.lock"})
TRUSTED_AUTHOR_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

GEMINI_REVIEW_MARKER = "Gemini Cherry-Pick Review Summary"
AUTOMATED_DECISION_PREFIX = "> **Automated Review Decision:"
SELF_REVIEW_REASON_SNIPPET = "reviewer token matches the PR author"
GUARDRAIL_BANNER_PREFIX = "> ⚠️ **Automated Guardrail:**"


def is_safe_repo_rel_path(path):
    """Returns True if path is a safe, normalized repository-relative file path.

    Rejects path traversal (`..`, `.`), `.git` segments, leading/trailing `/`,
    backslashes, ASCII control characters, backticks, `@{` sequences, segment-leading
    `@`, and leading `-` or `:` (which Git can interpret as flags or pathspec magic).
    """
    if not isinstance(path, str) or not path or len(path) > 1024:
        return False
    if "\\" in path or "`" in path or "@{" in path:
        return False
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        return False
    if path.startswith(("/", "-", ":")) or path.endswith("/"):
        return False
    for part in path.split("/"):
        if not part or part in (".", "..", ".git") or part.startswith("@"):
            return False
    return True


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
