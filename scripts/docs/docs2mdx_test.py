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


class Docs2MdxHeadingAnchorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "single_quotes",
          "<h2 id='common-attributes'>Attributes common to all build rules</h2>",
          "## Attributes common to all build rules {#common-attributes}",
      ),
      (
          "double_quotes",
          '<h2 id="typical-attributes">Typical attributes defined by most build rules</h2>',
          "## Typical attributes defined by most build rules {#typical-attributes}",
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


if __name__ == "__main__":
  unittest.main()
