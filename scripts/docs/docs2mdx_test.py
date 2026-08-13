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
          "[`--define=<a 'name=value' assignment>`](#build-flag--define)",
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
          "[`--test_string=<a string>`](#flag--test_string)",
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
          "[`--[no]expanded_a`](#flag--expanded_a)",
      ),
  )
  def test_flag_anchor_preserved(self, html, anchor_id, expected_link):
    result = docs2mdx._transform("test.html", html)
    self.assertIn(f'<a id="{anchor_id}"></a>', result)
    self.assertIn(expected_link, result)

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
