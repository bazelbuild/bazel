# Lint as: python3
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
"""Contains fixes for .mdx syntax errors in both generated and narrative docs.

This code is invoked for generated docs via docs2mdx.py, and for narrative docs
(static .mdx files) via create_release_docs.py.

Right now we need to process narrative docs since the 8.x and 9.x release
branches have several syntax errors in their .mdx files.
Once we move to Bazel 10 and beyond, this should no longer be necessary, which
will
allow us to move this code into docs2mdx.
"""

import re


def _fix_pre(m):
  """Converts a matched HTML <pre> block into a fenced Markdown code block."""
  prefix, content, suffix = m.groups()
  parts = [prefix]
  if prefix.strip():
    parts.append("\n")

  parts.append("```\n")
  parts.append(content.strip())  # TODO(fwe): Replace escaped HTML.
  parts.append("\n```")

  if suffix.strip():
    parts.append("\n")

  parts.append(suffix)
  return "".join(parts)


def _fix_link(m):
  raw = m.group(1)
  # Only keep .html extension for external links.
  if raw.startswith("http://") or raw.startswith("https://"):
    return m.group(0)

  return f"]({raw}"


def _fix_td_linebreaks(m):
  lb = "\n" if "\n" in m.group(1) else ""
  return f"<td>{lb}{m.group(1).strip()}{lb}</td>"


def _fix_title(m):
  title = m.group(2)
  quot = '"' if "'" in title else "'"
  return f"{m.group(1)}{quot}{title}{quot}"


# Restores MDX heading anchors escaped during markdown conversion.
_ESCAPED_HEADING_ANCHOR_SUB = (r" {#\1}", re.compile(r" &lcub;#([^&]+)&rcub;"))
# {: .external}, {:.devsite-disable-click-to-copy}
_TAG_SUB = ("", re.compile(r"\s*\{:\s?\.\S+\}"))
_DISABLE_FINDING_SUB = ("", re.compile(r"\{# disableFinding\([^)]+\) #\}"))
# {:#foo} and {: #foo } -> {#foo}
_KEYWORDS_SUB = ("", re.compile(r"^keywords: .+$", re.MULTILINE))
# https://github.com/bazelbuild/bazel/commit/6ec6d867843d274fa4555eb635ebafc60259b88e
_BAD_TITLE_SUB = (
    r"---\ntitle: \1\n---\n\n",
    re.compile(r"^---\n\n## title: (.+)\n\n", re.MULTILINE),
)
_PRE_BLOCK_SUB = (
    _fix_pre,
    re.compile(
        r"^([^\n]*?)(?:<pre[^>]*>)(.*?)(?:</pre>)([^\n]*?)$",
        re.DOTALL | re.MULTILINE,
    ),
)
# {{ '<var>' }} / {{ "</sub>" }} or any variations thereof
_DOUBLE_BRACKET_SUB = (r"\1", re.compile(r"\{\{ ['\"](</?\w+>)['\"] \}\}"))
_HTML_COMMENT_SUB = (r"{/* \1 */}", re.compile(r"<!--(.*?)-->", re.DOTALL))
_ANCHOR_SUB = (r"{\1}", re.compile(r"\{:\s?(#[\S+]+)\s?\}"))
_HTML_LINK_SUB = (_fix_link, re.compile(r"\]\(([^)]+)\.html"))
_ANGLE_BRACKET_LINK_SUB = (r"\1", re.compile(r"<(https?://[^>]+)>"))
# {# some comment #} -> {/* some comment */}
_BAD_COMMENT_SUB = (
    r"\1{/*\2*/}\3",
    re.compile(r"^(.*?)\{#(.*?)#\}(.*)$", re.MULTILINE),
)
_SELF_CLOSING_TAG_SUB = (
    r"<\1\2/>",
    re.compile(r"<(img|hr|col|br)\b([^>]*?)(/?)>"),
)
# ```--output package``` -> fenced block with the content on its own line
_SINGLE_LINE_FENCE_SUB = (
    r"\1```\n\1\2\n\1```",
    re.compile(r"^([ \t]*)```([^`\n]+)```[ \t]*$", re.MULTILINE),
)
_CLOSING_BR_SUB = ("", re.compile(r"</br>"))
_ORPHAN_CLOSING_P_SUB = ("", re.compile(r"\n\n</p>(?=\n+```)"))
_BAD_LINEBREAK_TD_SUB = (
    _fix_td_linebreaks,
    re.compile(r"<td>(.*?)</td>", re.DOTALL),
)
_BAD_LINEBREAK_P_SUB = (
    r"<p>\1</p>",
    re.compile(r"<p>([^\n]+(?:\n(?!\n)[^\n]+)*?)\n[ \t]*</p>"),
)
_ESCAPED_TD_BACKSLASH_SUB = (r"\\\\</td>", re.compile(r"(?<!\\)\\</td>"))
_UNESCAPED_INC_TAG_SUB = (r"&lt;inc\1&gt;", re.compile(r"<inc(\d+)>"))
_MISMATCHED_BACKTICK_QUOTE_SUB = (r"`\1`", re.compile(r"`(<[^>]+>)'"))
_UNESCAPED_DICT_BRACES_SUB = (
    r"&lcub;\1&rcub;",
    re.compile(r"\{(`[^`]+`:`[^`]+`)\}"),
)
_ALIGN_SUB = (
    r'\1"\2"',
    re.compile(r"(align=)(left|right|center|justify)"),
)  # There is only one match, so not very efficient.
_BOTTOM_NAV_SUB = (
    "",
    re.compile(
        r'^<table class="columns">.*?</table>$', re.MULTILINE | re.DOTALL
    ),
)
_TITLE_FIX_SUB = (
    _fix_title,
    re.compile(r"^(title: )'(.*?)( \{.*?\})?'$", re.MULTILINE),
)
_LEGACY_TAGS_SUB = ("", re.compile(r"^\s*</(body|html)>\s*$", re.MULTILINE))

_SUBS = [
    _ESCAPED_HEADING_ANCHOR_SUB,
    _TAG_SUB,
    _DISABLE_FINDING_SUB,
    _KEYWORDS_SUB,
    _BAD_TITLE_SUB,
    _PRE_BLOCK_SUB,
    _DOUBLE_BRACKET_SUB,
    _HTML_COMMENT_SUB,
    _ANCHOR_SUB,
    _HTML_LINK_SUB,
    _ANGLE_BRACKET_LINK_SUB,
    _BAD_COMMENT_SUB,
    _SELF_CLOSING_TAG_SUB,
    _SINGLE_LINE_FENCE_SUB,
    _CLOSING_BR_SUB,
    _ORPHAN_CLOSING_P_SUB,
    _BAD_LINEBREAK_TD_SUB,
    _BAD_LINEBREAK_P_SUB,
    _ESCAPED_TD_BACKSLASH_SUB,
    _UNESCAPED_INC_TAG_SUB,
    _MISMATCHED_BACKTICK_QUOTE_SUB,
    _UNESCAPED_DICT_BRACES_SUB,
    _ALIGN_SUB,
    _BOTTOM_NAV_SUB,
    _TITLE_FIX_SUB,
    _LEGACY_TAGS_SUB,
]

_HTML_STYLE_RE = re.compile(r"^</?style>", re.MULTILINE)
_TITLE_RE = re.compile(r"^title:\s*['\"]?", re.MULTILINE)
_MD_FRONT_MATTER_RE = re.compile(r"^---", re.MULTILINE)
_HEADING_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def apply(content):
  """Applies all MDX syntax and formatting fixes to the given document content."""
  fixed = _remove_trailing_whitespaces(content)
  fixed = (
      fixed
      if _TITLE_RE.search(fixed)
      else _HEADING_RE.sub(r"---\ntitle: '\1'\n---", fixed, count=1)
  )

  for sub, pattern in _SUBS:
    fixed = pattern.sub(sub, fixed)

  view_source = fixed.replace(
      "[View rule sourceopen_in_new]", "[View rule source]"
  )
  front_matter_first = _remove_anything_before_front_matter(view_source)
  return _remove_style_sections(front_matter_first)


def _remove_trailing_whitespaces(content):
  lines = (l.rstrip() for l in content.split("\n"))
  return "\n".join(lines)


def _remove_anything_before_front_matter(content):
  if content.startswith("---\n"):
    return content

  parts = _MD_FRONT_MATTER_RE.split(content, maxsplit=1)
  if len(parts) == 1:
    # Technically this only affects files that we need for the old site,
    # so the better solution would be to stop generating them.
    return parts[0]

  return f"---{parts[1]}"


def _remove_style_sections(content):
  m = _HTML_STYLE_RE.search(content)
  if not m:
    return content

  parts = _HTML_STYLE_RE.split(content)
  return f"{parts[0]}{parts[2].lstrip()}"
