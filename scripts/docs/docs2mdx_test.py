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


if __name__ == "__main__":
  unittest.main()
