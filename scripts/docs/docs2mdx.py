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
    '^((Project|Book):.+\n)', re.MULTILINE# | re.DOTALL
)
_TITLE_RE = re.compile(r"^title: '")

def _convert_directory(root_dir, mdx_dir):
  """Converts all .html and .md(x) files to .mdx files.

  Args:
      root_dir: str; full path of the directory with .html/.md(x)
        files (input).
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
  md = _html2md(content) if path.endswith(".html") else content
  return _fix_markdown(md)


def _html2md(content):
  return markdownify.markdownify(content, heading_style="ATX")


def _fix_markdown(content):
  no_tags = _TAG_RE.sub("", content)
  no_metadata = _METADATA_PATTERN.sub("", no_tags, count=2).lstrip()
  no_templates = _TEMPLATE_RE.sub("", no_metadata)
  no_html_links = _HTML_LINK_RE.sub(_fix_link, no_templates)
  fixed_headings = no_html_links if _TITLE_RE.search(no_html_links) else _HEADING_RE.sub("---\ntitle: '\\1'\n---", no_html_links, count=1)
  no_double_empty_lines = fixed_headings.replace("\n\n\n", "\n\n")
  return _remove_trailing_whitespaces(no_double_empty_lines)


def _remove_trailing_whitespaces(content):
  lines = (l.rstrip() for l in content.split("\n"))
  return "\n".join(lines)


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
