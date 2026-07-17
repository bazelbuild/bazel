# Copyright 2022 The Bazel Authors. All rights reserved.
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

import os
import unittest
from absl.testing import parameterized
from scripts.docs import rewriter


class CanRewriteTest(parameterized.TestCase):

  @parameterized.parameters(("/file/doc.md", True), ("/path/_book.yaml", True),
                            ("http://www.bazel.build/foo.html", True),
                            ("/dir/test.txt", False),
                            ("/images/aspects.svg", False))
  def testCanRewrite(self, path, expected_can_rewrite):
    self.assertEqual(rewriter.can_rewrite(path), expected_can_rewrite)


def read_data_file(basename, in_or_out_fragment):
  path = os.path.join(
      os.getenv("TEST_SRCDIR"),
      os.getenv("TEST_WORKSPACE"),
      "scripts/docs/testdata",
      in_or_out_fragment, basename)
  with open(path, "rt", encoding="utf-8") as f:
    return path, f.read()


class RewriteLinksTest(parameterized.TestCase):

  @parameterized.parameters(("_book.yaml"), ("doc.md"),
                            ("markdown_with_html.md"), ("site.html"),
                            ("yaml_with_html.yaml"))
  def testRewrite(self, basename):
    input_path, content = read_data_file(basename, "input")
    _, version = read_data_file("VERSION", "input")

    actual = rewriter.rewrite_links(input_path, content, basename, version)

    _, expected = read_data_file(basename, "expected_output")

    self.assertEqual(actual, expected)


if __name__ == "__main__":
  unittest.main()
