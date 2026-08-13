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

  def test_flag_docs_preserve_copyable_anchor_links(self):
    # Matches HtmlUtils.getUsageHtml() output from OptionsUsageTest.
    html = """<html><body>
<h1>Command-Line Reference</h1>
<dl>Startup options:
<dt id="flag--test_string"><code><a href="#flag--test_string">--test_string</a>=&lt;a string&gt;</code> default: "test string default"</dt>
<dd>
<p>a string-valued option to test simple option operations</p>
</dd>
<dt id="build-flag--jobs"><code id="jobs"><a href="#build-flag--jobs">--jobs</a>=&lt;an integer&gt;</code> default: "auto"</dt>
<dd>
<p>number of parallel jobs</p>
<p>Expands to:
<br/>&nbsp;&nbsp;<code><a href="#flag--local_test_jobs">--local_test_jobs=0</a></code>
</p></dd>
</dl>
</body></html>"""
    result = docs2mdx._transform("command-line-reference.html", html)

    self.assertIn('<a id="flag--test_string"></a>', result)
    self.assertIn(
        '[`--test_string=&lt;a string&gt;`](#flag--test_string) default: "test'
        ' string default"',
        result,
    )
    self.assertIn('<a id="build-flag--jobs"></a>', result)
    self.assertIn('[`--jobs=&lt;an integer&gt;`](#build-flag--jobs)', result)
    self.assertIn(
        '[`--local_test_jobs=0`](#flag--local_test_jobs)',
        result,
    )
    self.assertNotIn(
        '`--test_string=&lt;a string&gt;` default: "test string default"',
        result,
    )


if __name__ == "__main__":
  unittest.main()
