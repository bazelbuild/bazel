# Lint as: python3
# pylint: disable=g-direct-third-party-import
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
"""A tool for converting .html/.md(x) docs to valid .mdx files."""

import os
import pathlib
import re
import sys

from absl import app
from absl import flags
import markdownify


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "in_dir",
    None,
    "Absolute path of the input directory (where .html and .md(x) files "
    "should be read from).",
)
flags.DEFINE_string(
    "out_dir",
    None,
    "Absolute path of the output directory (where .mdx files should be"
    " written to).",
)
flags.mark_flag_as_required("in_dir")
flags.mark_flag_as_required("out_dir")


_HEADING_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_TEMPLATE_RE = re.compile(r"^\{%.+$\n", re.MULTILINE)
_TAG_RE = re.compile(r"\s?\{:[^}]+\}")
_HTML_LINK_RE = re.compile(r"\]\(([^)]+)\.html")
_METADATA_PATTERN = re.compile(
    "^((Project|Book):.+\n)", re.MULTILINE
)
_TITLE_RE = re.compile(r"^title: '", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ANGLE_BRACKET_LINK_RE = re.compile(r"<(https?://[^>]+)>")
_HTML_STYLE_PATTERN = re.compile(r"^</?style>", re.MULTILINE)
_MD_FRONT_MATTER_PATTERN = re.compile(r"^---", re.MULTILINE)
# Flag docs wrap the anchor link inside <code>, which markdownify drops.
# Move the link outside <code> so it survives conversion to MDX definition
# lists.
_CODE_FLAG_LINK_RE = re.compile(
    r'<code(?:\s[^>]*)?><a href="(#[^"]+)">(.*?)</a>(.*?)</code>',
    re.DOTALL,
)
# Definition-list flag terms that should expose a copyable deep-link anchor.
_FLAG_TERM_LINK_RE = re.compile(
    r"^\[`([^`]+)`\]\(#((?:[^)]*-)?flag--[^)]+)\)",
)
_HEADING_TAG_RE = re.compile(
    r"<h([1-6])([^>]*)>(.*?)</h\1>", re.DOTALL | re.IGNORECASE
)
_HEADING_ID_ATTR_RE = re.compile(r"""\bid=(["'])([^"']+)\1""")
_ESCAPED_HEADING_ANCHOR_RE = re.compile(r" &lcub;#([^&]+)&rcub;")

# In prose (outside code/pre blocks), these characters must be converted to
# HTML entities so they don't look like JSX or JavaScript blocks to MDX parsers.
_REPLACED_JS_CHARACTERS = {
    "{": "&lcub;",
    "}": "&rcub;",
}

_REPLACED_CODE_CHARACTERS = {
    "<": "&lt;",
    ">": "&gt;",
    **_REPLACED_JS_CHARACTERS,
}


def _escape_chars(text, replacements):
  """Escapes characters in a string.

  Args:
    text: str; string that needs characters escaped.
    replacements: dict[str, str]; a dictionary mapping characters to escape with
      their replacements.

  Returns:
    The escaped version of `text`.
  """
  for c in replacements.keys():
    text = text.replace(c, replacements[c])
  return text


# Table cells containing these elements cannot be represented as plain markdown
# table cell text. Preserve their inner HTML so MDX renders them correctly.
_COMPLEX_CELL_TAGS = frozenset(["ul", "ol", "table", "pre"])


def _cell_has_complex_content(cell):
  """Returns True if a table cell contains content that needs HTML preservation."""
  return cell.find(list(_COMPLEX_CELL_TAGS)) is not None


def _cell_inner_html(cell):
  """Returns the raw inner HTML of a table cell."""
  return "".join(str(child) for child in cell.children).strip()


def _format_table_cell(cell, content):
  """Formats table cell content as a markdown table cell."""
  colspan = 1
  if "colspan" in cell.attrs and cell["colspan"].isdigit():
    colspan = max(1, min(1000, int(cell["colspan"])))
  # Markdown table rows must be single-line; HTML in cells is fine on one line.
  return " " + content.replace("\n", " ") + " |" * colspan


class AcornSafeMarkdownConverter(markdownify.MarkdownConverter):
  """Custom converter that produces Acorn-parsable MDX output."""

  def convert_td(self, el, text, parent_tags):
    if _cell_has_complex_content(el):
      return _format_table_cell(el, _cell_inner_html(el))
    return super().convert_td(el, text, parent_tags)

  def convert_th(self, el, text, parent_tags):
    if _cell_has_complex_content(el):
      return _format_table_cell(el, _cell_inner_html(el))
    return super().convert_th(el, text, parent_tags)

  def convert_code(self, node, text, parent_tags):
    """Normalize whitespace in inline code before converting.

    Args:
      node: The HTML element being converted.
      text: The text content within the code tag.
      parent_tags: A list of parent tag names.

    Returns:
      The converted markdown string.
    """
    if "pre" not in parent_tags:
      # Multi-line <code> elements in the source HTML cause acorn parse errors
      # when curly braces span line boundaries. Collapsing whitespace first
      # lets the standard backtick conversion handle them on a single line.
      text = " ".join(text.split())

    return super().convert_code(node, text, parent_tags)

  def escape(self, text, parent_tags):
    """Custom escape handling."""
    if not text:
      return text
    escaped = super().escape(text, parent_tags)

    # Unescape underscores that are in the middle of words.
    escaped = re.sub(r"(\w)\\_(\w)", r"\1_\2", escaped)
    # Fenced and inline code blocks are already safe from MDX parsing.
    if "pre" in parent_tags or "code" in parent_tags:
      return escaped
    return _escape_chars(escaped, _REPLACED_CODE_CHARACTERS)


def _convert_directory(root_dir, mdx_dir):
  """Converts all .html and .md(x) files to .mdx files.

  Args:
      root_dir: str; full path of the directory with .html/.md(x) files (input).
      mdx_dir: str; full path of the directory where .mdx files should be
        created (output).
  """
  for curr_dir, _, files in os.walk(root_dir):
    rel = os.path.relpath(curr_dir, start=root_dir)
    dest_dir = os.path.join(mdx_dir, rel)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in files:
      basename, ext = os.path.splitext(fname)

      if ext not in (".html", ".md", ".mdx"):
        continue

      src = os.path.join(curr_dir, fname)
      dest = os.path.join(dest_dir, f"{basename}.mdx")
      _convert_file(src, dest)


def _convert_file(src, dest):
  with open(src, "rt") as f:
    content = f.read()

  with open(dest, "wt") as f:
    f.write(_transform(src, content))


def _transform(path, content):
  if path.endswith(".html"):
    md = _html2md(content)
  else:
    md = content
  return _post_markdown_transforms(md)


def _html2md(content):
  return AcornSafeMarkdownConverter(heading_style="ATX").convert(content)


def _pre_markdown_transforms(content):
  """Transforms applied to all sources before any markdown conversion.

  Args:
    content: str; content of an HTML or .md file.

  Returns:
    The file with invalid content removed.
  """
  no_tags = _TAG_RE.sub("", content)
  no_comments = _HTML_COMMENT_RE.sub("", no_tags)
  # Remove Project: and Book: lines
  no_metadata = _METADATA_PATTERN.sub("", no_comments, count=2).lstrip()
  no_templates = _TEMPLATE_RE.sub("", no_metadata)
  heading_anchors = _convert_heading_ids_to_mdx_anchors(no_templates)
  return _move_flag_links_outside_code(heading_anchors)


def _move_flag_links_outside_code(content):
  """Moves in-code flag anchor links outside of <code> tags.

  HtmlUtils.getUsageHtml() renders flags as
  <code><a href="#flag--name">--name</a>...</code>. Markdownify discards links
  nested inside inline code, so restructure the HTML before conversion.

  Args:
    content: str; HTML content before markdown conversion.

  Returns:
    Content with flag links moved outside of <code> tags.
  """
  return _CODE_FLAG_LINK_RE.sub(
      r'<a href="\1"><code>\2\3</code></a>',
      content,
  )


def _convert_heading_ids_to_mdx_anchors(content):
  """Converts HTML headings with id attributes to MDX anchor syntax.

  Example: <h2 id='foo'>Title</h2> -> ## Title {#foo}

  Headings without an id attribute are left unchanged for markdownify.

  Args:
    content: str; HTML content before markdown conversion.

  Returns:
    Content with id-bearing headings converted to MDX anchor syntax.
  """

  def repl(match):
    level = int(match.group(1))
    attrs = match.group(2)
    text = match.group(3).strip()
    id_match = _HEADING_ID_ATTR_RE.search(attrs)
    if not id_match:
      return match.group(0)
    heading_id = id_match.group(2)
    return f"{'#' * level} {text} {{#{heading_id}}}"

  return _HEADING_TAG_RE.sub(repl, content)


def _post_markdown_transforms(content):
  """Transforms applied to all sources after any markdown conversion.

  Args:
    content: str; content of a converted .mdx file.

  Returns:
    The content as fully valid .mdx.
  """
  no_html_links = _HTML_LINK_RE.sub(_fix_link, content)
  no_angle_links = _ANGLE_BRACKET_LINK_RE.sub(r"\1", no_html_links)
  no_double_empty_lines = no_angle_links.replace("\n\n\n", "\n\n")
  no_trailing_whitespaces = _remove_trailing_whitespaces(no_double_empty_lines)
  fixed_headings = (
      no_trailing_whitespaces
      if _TITLE_RE.search(no_trailing_whitespaces)
      else _HEADING_RE.sub(_fix_title_heading, no_trailing_whitespaces, count=1)
  )
  front_matter_first = _remove_anything_before_front_matter(fixed_headings)
  no_styles = _remove_style_sections(front_matter_first)
  restored_headings = _restore_heading_anchors(no_styles)
  return _add_flag_anchor_targets(restored_headings)


def _add_flag_anchor_targets(content):
  """Inserts explicit anchor targets for copyable per-flag deep links.

  After markdown conversion, flag terms look like
  [`--flag_name`](#flag--flag_name). Mintlify needs an element with a matching
  id attribute for those links (and copied URLs) to resolve.

  Args:
    content: str; MDX content after markdown conversion.

  Returns:
    Content with <a id="..."></a> targets inserted before each flag term.
  """
  seen_anchor_ids = set()
  lines = []
  for line in content.split("\n"):
    match = _FLAG_TERM_LINK_RE.match(line)
    if match:
      anchor_id = match.group(2)
      if anchor_id not in seen_anchor_ids:
        seen_anchor_ids.add(anchor_id)
        lines.append(f'<a id="{anchor_id}"></a>')
        lines.append("")
    lines.append(line)
  return "\n".join(lines)


def _restore_heading_anchors(content):
  """Restores MDX heading anchors escaped during markdown conversion."""
  return _ESCAPED_HEADING_ANCHOR_RE.sub(r" {#\1}", content)


def _remove_trailing_whitespaces(content):
  lines = (l.rstrip() for l in content.split("\n"))
  return "\n".join(lines)


def _fix_title_heading(m):
  title = m.group(1).replace("'", "\\'")
  return f"---\ntitle: '{title}'\n---"


def _remove_anything_before_front_matter(content):
  if content.startswith("---\n"):
    return content

  parts = _MD_FRONT_MATTER_PATTERN.split(content, maxsplit=1)
  if len(parts) == 1:
    # Technically this only affects files that we need for the old site,
    # so the better solution would be to stop generating them.
    return parts[0]

  return f"---{parts[1]}"


def _remove_style_sections(content):
  m = _HTML_STYLE_PATTERN.search(content)
  if not m:
    return content

  parts = _HTML_STYLE_PATTERN.split(content)
  return f"{parts[0]}{parts[2].lstrip()}"


def _fix_link(m):
  raw = m.group(1)
  # Only keep .html extension for external links.
  if raw.startswith("http://") or raw.startswith("https://"):
    return m.group(0)

  return f"]({raw}"


def _fail(msg):
  print(msg, file=sys.stderr)
  exit(1)


def main(unused_argv):
  if not os.path.isdir(FLAGS.in_dir):
    _fail(f"{FLAGS.in_dir} is not a directory")
  if not os.path.isdir(FLAGS.out_dir):
    _fail(f"{FLAGS.out_dir} is not a directory")

  _convert_directory(FLAGS.in_dir, FLAGS.out_dir)


if __name__ == "__main__":
  FLAGS(sys.argv)
  app.run(main)
