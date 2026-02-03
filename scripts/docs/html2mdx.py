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
"""A tool for converting .html docs to .mdx files."""
import os
import re
import sys

import markdownify

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "in_dir",
    None,
    "Absolute path of the input directory (where .html files should be read).",
)
flags.DEFINE_string(
    "out_dir",
    None,
    "Absolute path of the output directory (where .mdx files should be written).",
)
flags.mark_flag_as_required('in_dir')
flags.mark_flag_as_required('out_dir')


_HEADING_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_TEMPLATE_RE = re.compile(r"^\{%.+$\n", re.MULTILINE)
_HTML_LINK_RE = re.compile(r"\]\(([^)]+)\.html")


def _convert_directory(html_dir, mdx_dir):
    """Converts all .html files to .mdx files.
    
    Args:
        html_dir: str; full path of the directory with
            .html files (input).
        mdx_dir: str; full path of the directory where
            .mdx files should be created (output).
    """
    for curr_dir, _, files in os.walk(html_dir):
        rel = os.path.relpath(curr_dir, start=html_dir)
        dest_dir = os.path.join(mdx_dir, rel)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in files:
            basename, ext = os.path.splitext(fname)
            if ext != ".html":
                continue

            src = os.path.join(curr_dir, fname)
            dest = os.path.join(dest_dir, f"{basename}.mdx")
            
            _convert_file(src, dest)


def _convert_file(src, dest):
    with open(src, "rt") as f:
        content = f.read()
    
    with open(dest, "wt") as f:
        f.write(_transform(content))


def _transform(html_content):
    return _fix_markdown(_html2md(html_content))


def _html2md(content):
    return markdownify.markdownify(content, heading_style="ATX")


def _fix_markdown(content):
    no_templates = _TEMPLATE_RE.sub("", content)
    no_html_links = _HTML_LINK_RE.sub(_fix_link, no_templates)
    return _HEADING_RE.sub("---\ntitle: '\\1'\n---", no_html_links)


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
