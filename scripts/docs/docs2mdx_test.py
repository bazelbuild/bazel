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
  def testComplexTableCellContent(self, html, expected_substr, unexpected_substr):
    actual = docs2mdx._html2md(html)
    self.assertIn(expected_substr, actual)
    self.assertNotIn(unexpected_substr, actual)


if __name__ == "__main__":
  unittest.main()
