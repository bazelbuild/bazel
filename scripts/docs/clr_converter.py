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
"""Converts command-line-reference HTML to MDX with ParamField components."""

import dataclasses
import html
import html.parser
import re


@dataclasses.dataclass
class _Option:
  anchor_id: str = ""
  flag_name: str = ""
  value_type: str = ""
  abbreviation: str = ""
  default: str = ""
  allow_multiple: bool = False
  is_deprecated: bool = False
  help_html: str = ""
  expansion_flags: list = dataclasses.field(default_factory=list)
  tags_html: str = ""


@dataclasses.dataclass
class _Section:
  heading: str = ""
  anchor: str = ""
  inherits_html: str = ""
  categories: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Category:
  description: str = ""
  options: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _TagTable:
  heading: str = ""
  rows: list = dataclasses.field(default_factory=list)


_REPLACED_JS_CHARACTERS = {
    "{": "&lcub;",
    "}": "&rcub;",
}

_REPLACED_CODE_CHARACTERS = {
    "<": "&lt;",
    ">": "&gt;",
    **_REPLACED_JS_CHARACTERS,
}


def _escape_mdx(text):
  for c, replacement in _REPLACED_JS_CHARACTERS.items():
    text = text.replace(c, replacement)
  return text


def _escape_mdx_code(text):
  # Escape $ so MDX doesn't parse it as a LaTeX math delimiter or,
  # combined with {, as a JS template expression.
  text = text.replace("$", "&#36;")
  for c, replacement in _REPLACED_CODE_CHARACTERS.items():
    text = text.replace(c, replacement)
  return text


class _CLRParser(html.parser.HTMLParser):
  """Parses the command-line-reference HTML into structured data."""

  def __init__(self):
    super().__init__()
    self.prefix_html = ""
    self.commands_table_rows = []
    self.sections = []
    self.tag_tables = []

    self._state = "PREFIX"
    self._depth = 0
    self._tag_stack = []

    self._current_section = None
    self._current_category = None
    self._current_option = None
    self._current_tag_table = None

    self._in_dt = False
    self._in_dd = False
    self._in_h2 = False
    self._in_h3 = False
    self._in_commands_table = False
    self._in_tag_table = False
    self._in_table_row = False
    self._table_cells = []
    self._current_cell_html = ""
    self._in_table_cell = False
    self._in_dl = False
    self._dl_text_buffer = ""
    self._inherits_buffer = ""

    self._dt_html = ""
    self._dd_html = ""
    self._heading_text = ""
    self._heading_anchor = ""

    self._prefix_parts = []
    self._saw_commands_heading = False

  def handle_starttag(self, tag, attrs):
    attrs_dict = dict(attrs)

    if self._state == "PREFIX":
      if tag == "h2":
        self._in_h2 = True
        self._heading_text = ""
        return
      self._prefix_parts.append(self._rebuild_tag(tag, attrs))
      return

    if self._in_dt:
      self._dt_html += self._rebuild_tag(tag, attrs)
      return

    if self._in_dd:
      self._dd_html += self._rebuild_tag(tag, attrs)
      return

    if self._in_table_cell:
      self._current_cell_html += self._rebuild_tag(tag, attrs)
      return

    if tag == "h2":
      self._in_h2 = True
      self._heading_text = ""
      self._heading_anchor = ""
      return

    if tag == "h3":
      self._in_h3 = True
      self._heading_text = ""
      return

    if tag == "a" and self._in_h2:
      name = attrs_dict.get("name", "")
      if name:
        self._heading_anchor = name
      return

    if tag == "table":
      if self._in_h3 or (self._current_tag_table is not None):
        self._in_tag_table = True
      elif self._state == "COMMANDS_TABLE":
        self._in_commands_table = True
      return

    if tag == "tr" and (self._in_commands_table or self._in_tag_table):
      self._in_table_row = True
      self._table_cells = []
      return

    if tag == "td" and self._in_table_row:
      self._in_table_cell = True
      self._current_cell_html = ""
      cell_id = attrs_dict.get("id", "")
      if cell_id:
        self._current_cell_html += f'<span id="{cell_id}">'
      return

    if tag == "p" and self._current_section and not self._in_dl:
      self._state = "INHERITS_P"
      self._inherits_buffer = ""
      return

    if tag == "a" and self._state == "INHERITS_P":
      self._inherits_buffer += self._rebuild_tag(tag, attrs)
      return

    if tag == "dl":
      self._in_dl = True
      self._dl_text_buffer = ""
      return

    if tag == "dt":
      self._in_dt = True
      self._dt_html = ""
      anchor_id = attrs_dict.get("id", "")
      self._current_option = _Option(anchor_id=anchor_id)
      return

    if tag == "dd":
      self._in_dd = True
      self._dd_html = ""
      return

    if tag == "p" and self._in_dl and not self._in_dt and not self._in_dd:
      self._in_dd = True
      self._dd_html = "<p>"
      return

  def handle_endtag(self, tag):
    if self._state == "PREFIX":
      if tag == "h2" and self._in_h2:
        self._in_h2 = False
        if self._heading_text.strip() == "Commands":
          self.prefix_html = "".join(self._prefix_parts)
          self._state = "COMMANDS_TABLE"
        else:
          self._prefix_parts.append(f"<h2>{self._heading_text}</h2>")
        return
      self._prefix_parts.append(f"</{tag}>")
      return

    if self._in_dt:
      if tag == "dt":
        self._in_dt = False
        self._parse_dt(self._dt_html)
        return
      self._dt_html += f"</{tag}>"
      return

    if self._in_dd:
      if tag == "dd":
        self._in_dd = False
        self._parse_dd(self._dd_html)
        if self._current_option and self._current_category:
          self._current_category.options.append(self._current_option)
          self._current_option = None
        return
      self._dd_html += f"</{tag}>"
      return

    if self._in_table_cell:
      if tag == "td":
        self._in_table_cell = False
        self._table_cells.append(self._current_cell_html)
        return
      self._current_cell_html += f"</{tag}>"
      return

    if tag == "tr" and self._in_table_row:
      self._in_table_row = False
      if self._in_commands_table:
        self.commands_table_rows.append(self._table_cells)
      elif self._in_tag_table and self._current_tag_table:
        self._current_tag_table.rows.append(self._table_cells)
      return

    if tag == "a" and self._state == "INHERITS_P":
      self._inherits_buffer += "</a>"
      return

    if tag == "p" and self._state == "INHERITS_P":
      if self._current_section:
        self._current_section.inherits_html = self._inherits_buffer
      self._state = "BODY"
      return

    if tag == "table":
      if self._in_commands_table:
        self._in_commands_table = False
        self._state = "BODY"
      elif self._in_tag_table:
        self._in_tag_table = False
        if self._current_tag_table:
          self.tag_tables.append(self._current_tag_table)
          self._current_tag_table = None
      return

    if tag == "h2" and self._in_h2:
      self._in_h2 = False
      self._finish_current_section()
      self._current_section = _Section(
          heading=self._heading_text.strip(),
          anchor=self._heading_anchor,
      )
      return

    if tag == "h3" and self._in_h3:
      self._in_h3 = False
      heading = self._heading_text.strip()
      if "Option" in heading and "Tag" in heading:
        self._current_tag_table = _TagTable(heading=heading)
      return

    if tag == "dl":
      self._in_dl = False
      if self._current_category:
        if self._current_section:
          self._current_section.categories.append(self._current_category)
        self._current_category = None
      return

    if tag == "p" and self._in_dl and not self._in_dt and not self._in_dd:
      pass

  def handle_data(self, data):
    if self._state == "PREFIX":
      if self._in_h2:
        self._heading_text += data
      else:
        self._prefix_parts.append(html.escape(data).replace("&#x27;", "'"))
      return

    if self._in_h2 or self._in_h3:
      self._heading_text += data
      return

    if self._in_dt:
      self._dt_html += html.escape(data)
      return

    if self._in_dd:
      self._dd_html += html.escape(data)
      return

    if self._in_table_cell:
      self._current_cell_html += html.escape(data)
      return

    if self._in_dl and not self._in_dt and not self._in_dd:
      stripped = data.strip().rstrip(":")
      if stripped:
        self._current_category = _Category(description=html.unescape(stripped))
      return

    if self._state == "INHERITS_P":
      self._inherits_buffer += html.escape(data)
      return

  def handle_entityref(self, name):
    text = f"&{name};"
    if self._in_dt:
      self._dt_html += text
    elif self._in_dd:
      self._dd_html += text
    elif self._in_table_cell:
      self._current_cell_html += text
    elif self._state == "INHERITS_P":
      self._inherits_buffer += text
    elif self._state == "PREFIX" and not self._in_h2:
      self._prefix_parts.append(text)

  def handle_charref(self, name):
    text = f"&#{name};"
    if self._in_dt:
      self._dt_html += text
    elif self._in_dd:
      self._dd_html += text
    elif self._in_table_cell:
      self._current_cell_html += text

  def close(self):
    super().close()
    self._finish_current_section()

  def _finish_current_section(self):
    if self._current_category and self._current_section:
      self._current_section.categories.append(self._current_category)
      self._current_category = None
    if self._current_section:
      self.sections.append(self._current_section)
      self._current_section = None

  def _rebuild_tag(self, tag, attrs):
    parts = [tag]
    for k, v in attrs:
      if v is None:
        parts.append(k)
      else:
        parts.append(f'{k}="{v}"')
    return "<" + " ".join(parts) + ">"

  def _parse_dt(self, dt_html):
    opt = self._current_option
    if not opt:
      return

    flag_match = re.search(r'<a[^>]*>([^<]+)</a>', dt_html)
    if flag_match:
      opt.flag_name = flag_match.group(1)

    code_match = re.search(r'<code[^>]*>(.*?)</code>', dt_html, re.DOTALL)
    if code_match:
      code_content = code_match.group(1)
      after_a = re.split(r'</a>', code_content, maxsplit=1)
      if len(after_a) > 1:
        value_part = after_a[1].strip()
        if value_part.startswith("="):
          raw_type = html.unescape(value_part[1:]).strip()
          if raw_type.startswith("<") and raw_type.endswith(">"):
            raw_type = raw_type[1:-1]
          opt.value_type = raw_type

    after_code = re.split(r'</code>\s*', dt_html, maxsplit=1)
    if len(after_code) > 1:
      remainder = after_code[1]

      abbrev_match = re.search(r'\[<code>(-\w)</code>\]', remainder)
      if abbrev_match:
        opt.abbreviation = abbrev_match.group(1)
        remainder = remainder[abbrev_match.end():]

      remainder = remainder.strip()
      if "multiple uses are accumulated" in remainder:
        opt.allow_multiple = True
      elif remainder.startswith("default:"):
        default_val = remainder[len("default:"):].strip()
        default_val = html.unescape(default_val).strip('"')
        opt.default = default_val

    if not opt.value_type and not opt.flag_name.startswith("--[no]"):
      opt.value_type = "void"
    elif not opt.value_type and opt.flag_name.startswith("--[no]"):
      opt.value_type = "boolean"

  def _parse_dd(self, dd_html):
    opt = self._current_option
    if not opt:
      return

    parts = re.split(r'<p>', dd_html)
    help_parts = []
    for part in parts:
      part_text = part.strip()
      if not part_text:
        continue
      if part_text.startswith("Expands to:"):
        exp_flags = re.findall(
            r'<code><a[^>]*>([^<]+)</a></code>', part_text)
        opt.expansion_flags = exp_flags
      elif part_text.startswith("Tags:"):
        opt.tags_html = "<p>" + part_text
        if "deprecated" in part_text.lower():
          if "metadata_tag_DEPRECATED" in part_text:
            opt.is_deprecated = True
      else:
        help_parts.append(part_text.removesuffix("</p>"))

    opt.help_html = "\n".join(help_parts)


def _escape_mdx_outside_code(text):
  """Escapes MDX-special characters outside of code spans and fenced blocks."""
  # First split on fenced code blocks (``` ... ```), then within non-fenced
  # segments split on inline backtick spans.
  fenced_parts = re.split(r'(```[^`]*```)', text, flags=re.DOTALL)
  result = []
  for fi, fenced_part in enumerate(fenced_parts):
    if fi % 2 == 1:
      result.append(fenced_part)
      continue
    inline_parts = re.split(r'(`[^`]*`)', fenced_part)
    for ii, inline_part in enumerate(inline_parts):
      if ii % 2 == 0:
        inline_part = _escape_mdx_code(inline_part)
      result.append(inline_part)
  return "".join(result)


_PRE_CODE_RE = re.compile(
    r'<pre><code>(.*?)</code></pre>', re.DOTALL)


def _convert_pre_code_block(m):
  content = html.unescape(m.group(1)).strip('\n')
  return f'\n\n```\n{content}\n```\n\n'


_ERRATA = [
    ("`rewrite, allow, block'", "`rewrite, allow, block`"),
]


def _apply_known_errata(text):
  for old, new in _ERRATA:
    text = text.replace(old, new)
  return text


def _html_to_simple_md(html_content):
  """Minimal HTML-to-markdown for option help text."""
  text = html_content

  # Fenced code blocks before inline code so <pre><code> isn't consumed
  # by the inline pattern. Content is unescaped but not MDX-escaped since
  # fenced blocks are literal in MDX.
  text = _PRE_CODE_RE.sub(_convert_pre_code_block, text)

  text = re.sub(r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>', r'[\2](\1)', text)
  text = re.sub(r'<code>([^<]*)</code>', lambda m: '`' + m.group(1) + '`', text)
  text = re.sub(r'<em>([^<]*)</em>', r'*\1*', text)
  text = re.sub(r'<strong>([^<]*)</strong>', r'**\1**', text)
  text = re.sub(r'<br\s*/?>', '\n', text)

  text = re.sub(r'<li>\s*', '- ', text)
  text = re.sub(r'</li>', '', text)
  text = re.sub(r'</?[uo]l>', '', text)

  text = re.sub(r'<p>', '\n\n', text)
  text = re.sub(r'</p>', '', text)

  text = re.sub(r'<[^>]+>', '', text)

  text = html.unescape(text)
  text = _apply_known_errata(text)
  text = _escape_mdx_outside_code(text)
  text = re.sub(r'\n{3,}', '\n\n', text)
  return text.strip()


def _escape_attr(value):
  """Escapes a string for use inside a JSX attribute value."""
  escaped = value.replace("&", "&amp;")
  escaped = escaped.replace('"', "&quot;")
  escaped = escaped.replace("<", "&lt;")
  escaped = escaped.replace(">", "&gt;")
  escaped = escaped.replace("{", "&lcub;")
  escaped = escaped.replace("}", "&rcub;")
  return escaped


def _render_option(opt):
  """Renders a single option as a ParamField component."""
  attrs = [f'path="{_escape_attr(opt.flag_name)}"']

  if opt.value_type:
    attrs.append(f'type="{_escape_attr(opt.value_type)}"')

  if opt.default and not opt.allow_multiple:
    attrs.append(f'default="{_escape_attr(opt.default)}"')

  if opt.is_deprecated:
    attrs.append("deprecated")

  attr_str = " ".join(attrs)
  lines = [f"<ParamField {attr_str}>"]

  if opt.abbreviation:
    lines.append(f"  Short form: `{opt.abbreviation}`")
    lines.append("")

  help_md = _html_to_simple_md(opt.help_html)
  if help_md:
    for line in help_md.split("\n"):
      lines.append(f"  {line}" if line.strip() else "")

  if opt.allow_multiple:
    lines.append("")
    lines.append("  *May be used multiple times; values are accumulated.*")

  if opt.expansion_flags:
    lines.append("")
    lines.append("  Expands to:")
    for flag in opt.expansion_flags:
      lines.append(f"  - `{_escape_mdx(flag)}`")

  if opt.tags_html:
    tags_md = _render_tags(opt.tags_html)
    if tags_md:
      lines.append("")
      lines.append(f"  {tags_md}")

  lines.append("</ParamField>")
  return "\n".join(lines)


def _render_tags(tags_html):
  """Converts tags HTML to markdown."""
  tag_links = re.findall(
      r'<a\s+href="([^"]*)"><code>([^<]*)</code></a>', tags_html)
  if not tag_links:
    return ""
  parts = [f"[`{name}`]({href})" for href, name in tag_links]
  return "Tags: " + ", ".join(parts)


def _render_commands_table(rows):
  """Renders the commands table as markdown."""
  lines = [
      "## Commands",
      "",
      "| | |",
      "| --- | --- |",
  ]
  for cells in rows:
    if len(cells) >= 2:
      cmd_md = re.sub(
          r'<a\s+href="([^"]*)"><code>([^<]*)</code></a>',
          lambda m: f"[`{m.group(2)}`]({m.group(1)})",
          cells[0],
      )
      desc = html.unescape(re.sub(r'<[^>]+>', '', cells[1]))
      lines.append(f"| {cmd_md} | {_escape_mdx(desc)} |")
  return "\n".join(lines)


def _render_tag_table(tag_table):
  """Renders a tag description table as markdown."""
  lines = [
      f"### {tag_table.heading}",
      "",
      "| | |",
      "| --- | --- |",
  ]
  for cells in tag_table.rows:
    if len(cells) >= 2:
      name_cell = cells[0]
      span_match = re.search(r'<span id="([^"]*)">', name_cell)
      code_match = re.search(r'<code>([^<]*)</code>', name_cell)
      if span_match and code_match:
        tag_id = span_match.group(1)
        tag_name = code_match.group(1)
        name_md = f'<span id="{tag_id}">`{tag_name}`</span>'
      else:
        name_md = re.sub(r'<[^>]+>', '', name_cell)
      desc = html.unescape(re.sub(r'<[^>]+>', '', cells[1]))
      lines.append(f"| {name_md} | {_escape_mdx(desc)} |")
  return "\n".join(lines)


def _render_section(section):
  """Renders a section heading, inherits info, and all options."""
  lines = [f"## {section.heading}"]
  if section.anchor:
    lines[0] = f'## {section.heading} {{#{section.anchor}}}'

  if section.inherits_html:
    inherits_md = re.sub(
        r'<a\s+href="([^"]*)">([^<]*)</a>',
        lambda m: f"[{m.group(2)}]({m.group(1)})",
        section.inherits_html,
    )
    inherits_md = re.sub(r'<[^>]+>', '', inherits_md).strip()
    if inherits_md:
      lines.append("")
      lines.append(inherits_md)

  for cat in section.categories:
    lines.append("")
    if cat.description:
      lines.append(cat.description)
    lines.append("")
    for opt in cat.options:
      lines.append(_render_option(opt))
      lines.append("")

  return "\n".join(lines)


def convert(html_content):
  """Converts command-line-reference HTML to MDX with ParamField components.

  Args:
    html_content: str; the full HTML content of command-line-reference.html.

  Returns:
    The MDX content with ParamField components.
  """
  parser = _CLRParser()
  parser.feed(html_content)
  parser.close()

  parts = []

  parts.append(_render_commands_table(parser.commands_table_rows))
  parts.append("")

  for section in parser.sections:
    parts.append(_render_section(section))
    parts.append("")

  for tag_table in parser.tag_tables:
    parts.append(_render_tag_table(tag_table))
    parts.append("")

  return "\n".join(parts)
