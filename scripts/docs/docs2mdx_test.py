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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import parameterized
from scripts.docs import docs2mdx


class Docs2MdxTest(unittest.TestCase):

  def test_code_blocks_keep_literal_characters(self):
    html = """<html><body>
<h1>Example</h1>
<pre><code>values = {"define": "species=excelsior"}
if x &lt; y { return true; }</code></pre>
</body></html>"""
    result = docs2mdx._transform("example.html", html)

    self.assertIn('values = {"define": "species=excelsior"}', result)
    self.assertNotIn("&lcub;", result)
    self.assertNotIn("&rcub;", result)
    code_block = result.split("```")[1]
    self.assertNotIn("&lt;", code_block)

  def test_prose_still_escapes_mdx_special_characters(self):
    html = """<html><body>
<p>Compare x &lt; y and use {braces} in prose.</p>
</body></html>"""
    result = docs2mdx._transform("example.html", html)

    self.assertIn("&lt;", result)
    self.assertIn("&lcub;", result)
    self.assertIn("&rcub;", result)

  def test_pre_blocks_in_markdown_are_not_entity_escaped(self):
    md = """# Title

<pre>
config_setting(
    values = {"define": "species=excelsior"},
)
</pre>
"""
    result = docs2mdx._transform("example.md", md)

    self.assertIn('values = {"define": "species=excelsior"}', result)
    self.assertNotIn("&lcub;", result)
    self.assertNotIn("&rcub;", result)


class Docs2MdxTableCellTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "list_in_table_cell",
          """
<table>
  <tr>
    <td><code>aspect_hints</code></td>
    <td>
      <p>Some text.</p>
      <p>Best practices:</p>
      <ul>
        <li>Targets listed in aspect_hints should be lightweight.</li>
        <li>Language-specific logic should consider only aspect hints.</li>
      </ul>
    </td>
  </tr>
</table>
""",
          "<ul>",
          "* Targets listed",
      ),
      (
          "nested_table_in_cell",
          """
<table>
  <tr>
    <td><code>size</code></td>
    <td>
      <p>Test sizes:</p>
      <table>
        <tr><th>Size</th><th>RAM (in MB)</th></tr>
        <tr><td>small</td><td>20</td></tr>
      </table>
    </td>
  </tr>
</table>
""",
          "<table>",
          "| Size | RAM",
      ),
      (
          "pre_in_table_cell",
          """
<table>
  <tr><th>Attribute</th><th>Description</th></tr>
  <tr>
    <td><code>url</code></td>
    <td>
      <p>URL of the file.</p>
      <pre><code>https://example.com/file.tar.gz</code></pre>
    </td>
  </tr>
</table>
""",
          "<pre><code>",
          "```",
      ),
  )
  def testComplexTableCellContent(
      self, html, expected_substr, unexpected_substr
  ):
    actual = docs2mdx._html2md(html)
    self.assertIn(expected_substr, actual)
    self.assertNotIn(unexpected_substr, actual)


class Docs2MdxHeadingAnchorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "single_quotes",
          (
              "<h2 id='common-attributes'>Attributes common to all build"
              " rules</h2>"
          ),
          "## Attributes common to all build rules {#common-attributes}",
      ),
      (
          "double_quotes",
          (
              '<h2 id="typical-attributes">Typical attributes defined by most'
              " build rules</h2>"
          ),
          (
              "## Typical attributes defined by most build rules"
              " {#typical-attributes}"
          ),
      ),
      (
          "extra_attributes",
          '<h2 id="cc_binary" class="deprecated">\n    cc_binary\n  </h2>',
          "## cc_binary {#cc_binary}",
      ),
      (
          "h3_heading",
          '<h3 id="package_args">Arguments</h3>',
          "### Arguments {#package_args}",
      ),
  )
  def test_heading_id_preserved(self, html, expected_heading):
    result = docs2mdx._transform("test.html", html)
    self.assertIn(expected_heading, result)

  def test_heading_without_id_has_no_anchor(self):
    html = "<h2>Rules</h2>"
    result = docs2mdx._transform("test.html", html)
    self.assertIn("## Rules", result)
    self.assertNotIn("{#", result)


class Docs2MdxFlagAnchorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "command_specific_flag",
          """
<dl>
<dt id="build-flag--define"><code id="define"><a href="#build-flag--define">--define</a>=&lt;a 'name=value' assignment&gt;</code> multiple uses are accumulated</dt>
<dd><p>Each --define option specifies an assignment for a build variable.</p></dd>
</dl>
""",
          "build-flag--define",
      ),
      (
          "global_flag",
          """
<dl>
<dt id="flag--test_string"><code><a href="#flag--test_string">--test_string</a>=&lt;a string&gt;</code> default: "test string default"</dt>
<dd><p>a string-valued option to test simple option operations</p></dd>
</dl>
""",
          "flag--test_string",
      ),
      (
          "boolean_flag",
          """
<dl>
<dt id="flag--expanded_a"><code><a href="#flag--expanded_a">--[no]expanded_a</a></code> default: "true"</dt>
<dd><p>boolean option</p></dd>
</dl>
""",
          "flag--expanded_a",
      ),
  )
  def test_flag_anchor_preserved(self, html, anchor_id):
    result = docs2mdx._transform("test.html", html)
    self.assertIn(f'<a id="{anchor_id}"></a>', result)
    self.assertIn(f"](#{anchor_id})", result)

  def test_duplicate_anchor_id_inserted_once(self):
    html = """
<dl>
<dt id="build-flag--define"><code><a href="#build-flag--define">--define</a>=&lt;value&gt;</code></dt>
<dd><p>First definition.</p></dd>
<dt id="build-flag--define"><code><a href="#build-flag--define">--define</a>=&lt;value&gt;</code></dt>
<dd><p>Duplicate id should not add a second anchor.</p></dd>
</dl>
"""
    result = docs2mdx._transform("test.html", html)
    self.assertEqual(result.count('<a id="build-flag--define"></a>'), 1)


if __name__ == "__main__":
  unittest.main()
