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
import re
import sys

from absl import app
from absl import flags
import markdownify
from scripts.docs import clr_converter
from scripts.docs import mdx_fixes


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


_TEMPLATE_RE = re.compile(r"^\{%.+$\n", re.MULTILINE)
_TAG_RE = re.compile(r"\s?\{:[^}]+\}")
# Kramdown IDs at the end of a list item: "*   [Foo](url){:#foo}".
_KRAMDOWN_LIST_ITEM_ID_RE = re.compile(
    r"^([ \t]*(?:[*+-]|\d+\.)[ \t]+.*?)\{:[ \t]?#([^\s}]+)[ \t]?\}[ \t]*$",
    re.MULTILINE,
)
_METADATA_PATTERN = re.compile(
    "^((Project|Book):.+\n)", re.MULTILINE
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
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
    r"^([^\n]*)<h([1-6])([^>]*)>(.*?)</h\2>",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_HEADING_ID_ATTR_RE = re.compile(r"""\bid=(["'])([^"']+)\1""")
# See docstrings of _format_table_cell() and ()
# for an explanation.
_TAGS_TO_FLATTEN_RE = re.compile(r"(?:</?(?:pre|code)[^>]*>)+")

# In prose (outside code/pre blocks), these characters must be converted to
# HTML entities so they don't look like JSX or JavaScript blocks to MDX parsers.
_REPLACED_JS_CHARACTERS = {
    "{": "&lcub;",
    "}": "&rcub;",
    "$": "&#36;",
}

_REPLACED_CODE_CHARACTERS = {
    "<": "&lt;",
    ">": "&gt;",
    **_REPLACED_JS_CHARACTERS,
}

_CONFIGURATION_HTML_BAD_LINE = re.compile(
    r"^<p>Use this to distinguish different configurations for the same"
    r" target.+$",
    re.MULTILINE,
)


def _escape_chars(text, replacements):
  """Escapes characters in a string.

  Args:
    text: str; string that needs characters escaped.
    replacements: dict[str, str]; a dictionary mapping characters to escape with
      their replacements.

  Returns:
    The escaped version of `text`.
  """
  for c in replacements:
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
  """Formats table cell content as a markdown table cell.

  Markdownify converts elements bottom-up.
  For table cells we discard the already converted tag and instead
  manually "convert" the inner HTML, which can be very complex.

  A better approach would be this:
  For every tag <foo> that can produce multi-line Markdown output:r
  If convert_foo is called for a nested table, it should preserve <foo>
  tags and return the (already converted) content as a single line.
  Otherwise it just delegates to the super class implementation.
  However, this cannot be implemented since parent_tags is a set and not a
  dict or multiset, so we cannot distinguish a simple table from a
  nested table.

  Args:
    cell: The BeautifulSoup table cell element (`td` or `th`).
    content: str; the raw inner HTML content of the table cell.

  Returns:
    The formatted single-line Markdown table cell string.
  """
  colspan = 1
  if "colspan" in cell.attrs and cell["colspan"].isdigit():
    colspan = max(1, min(1000, int(cell["colspan"])))
  # Content = raw HTML, i.e. it can contain tags such as <code> or <pre>
  # with forbidden characters in their values (such as curly braces).
  # Consequently, we need to escape them here.
  return f" {_convert_html_to_single_line_md(content)}{' |' * colspan}"


def _convert_html_to_single_line_md(content):
  """Converts 'bad' tags (<pre>, <code>) and fits the result into a single line.

  These tags are 'bad' since they can contain reserved chars such as curly
  braces,
  which lead to syntax errors if they appear outside of fenced Markdown code
  blocks.

  Args:
    content: str; the raw HTML content to convert.

  Returns:
    A single-line Markdown/HTML string with `<pre>` and `<code>` tags flattened.
  """
  # Convert <pre> and <code> to single fenced code blocks.
  no_bad_tags = _TAGS_TO_FLATTEN_RE.sub("`", content)

  # Escape special characters outside of fenced code blocks.
  parts = no_bad_tags.split("`")
  for i in range(0, len(parts), 2):
    parts[i] = _escape_chars(parts[i], _REPLACED_JS_CHARACTERS)
  escaped = "`".join(parts)

  # Remove line breaks.
  raw_lines = [l.strip() for l in escaped.split("\n")]
  return " ".join([l for l in raw_lines if l])


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
  """Transforms the content of an HTML or Markdown file into valid MDX."""
  content = _pre_markdown_transforms(content)
  if path.endswith(".html"):
    if os.path.basename(path) == "command-line-reference.html":
      md = clr_converter.convert(content)
    else:
      if path.endswith("configuration.html"):
        fixed_content = _fix_configuration_dot_html(content)
      elif path.endswith("bzl.html"):
        fixed_content = _fix_bzl_dot_html(content)
      else:
        fixed_content = content

      md = _html2md(fixed_content)
  else:
    md = content
  return _post_markdown_transforms(md)


def _fix_configuration_dot_html(content):
  """Fixes malformed HTML in rules/lib/builtins/configurations.html."""

  def fix(m):
    return f"{m.group(0).replace('.', '.</li>')}</ul></p>"

  return _CONFIGURATION_HTML_BAD_LINE.sub(fix, content)


def _fix_bzl_dot_html(content):
  """Fixes malformed HTML link in rules/lib/globals/bzl.html.

  There is only a single instance of this bug, so it doesn't
  make sense to implement a general solution.

  Args:
    content: str; the HTML content to be fixed.
  Returns:
    The fixed HTML content, as string.
  """
  href = "../globals/workspace#register_execution_platforms"
  return content.replace(f'"{href}>', f'"{href}">')


def _html2md(content):
  # HTML content needs a few extra transforms.
  content = _move_flag_links_outside_code(content)
  return AcornSafeMarkdownConverter(heading_style="ATX").convert(content)


def _pre_markdown_transforms(content):
  """Transforms applied to all sources before any markdown conversion.

  Args:
    content: str; content of an HTML or .md file.

  Returns:
    The file with invalid content removed.
  """
  no_tags = _TAG_RE.sub("", _convert_kramdown_list_item_ids(content))
  no_comments = _HTML_COMMENT_RE.sub("", no_tags)
  # Remove Project: and Book: lines
  no_metadata = _METADATA_PATTERN.sub("", no_comments, count=2).lstrip()
  no_templates = _TEMPLATE_RE.sub("", no_metadata)
  return _convert_heading_ids_to_mdx_anchors(no_templates)


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


def _convert_kramdown_list_item_ids(content):
  """Converts Kramdown IDs on list items in Markdown to HTML anchors.

  MDX anchor syntax ({#foo}) is only valid for headings, so list items get an
  explicit anchor element instead.

  Example: *   [Foo](url){:#foo} -> *   [Foo](url)<a name="foo"></a>

  This has to run before _TAG_RE removes all remaining {:...} attribute lists.

  Args:
    content: str; content of an HTML or .md file.

  Returns:
    Content with Kramdown list item IDs converted to anchor elements.
  """
  return _KRAMDOWN_LIST_ITEM_ID_RE.sub(r'\1<a name="\2"></a>', content)


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
    level = int(match.group(2))
    attrs = match.group(3)
    text = match.group(4).strip()
    id_match = _HEADING_ID_ATTR_RE.search(attrs)
    if not id_match:
      return match.group(0)
    # Hack: do not add anchor if heading has non-empty prefix (e.g. list tag)
    line_prefix = match.group(1).strip()
    heading_id = id_match.group(2)
    anchor = "" if line_prefix else f" {{#{heading_id}}}"
    return f"{'#' * level} {text}{anchor}"

  return _HEADING_TAG_RE.sub(repl, content)


def _post_markdown_transforms(content):
  """Transforms applied to all sources after any markdown conversion.

  Args:
    content: str; content of a converted .mdx file.

  Returns:
    The content as fully valid .mdx.
  """
  return _add_flag_anchor_targets(mdx_fixes.apply(content))


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
