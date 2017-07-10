# pylint: disable=g-bad-file-header
# Copyright 2017 The Bazel Authors. All rights reserved.
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

import os
import unittest
from src.test.py.bazel import test_base


class BazelExternalRepositoryTest(test_base.TestBase):

  def testNewHttpArchive(self):
    rule_definition = [
        'new_http_archive(',
        '    name = "six_archive",',
        '    urls = [',
        '      "http://mirror.bazel.build/pypi.python.org/%s' %
        'packages/source/s/six/six-1.10.0.tar.gz",',
        '      "https://pypi.python.org/packages/%s' %
        'source/s/six/six-1.10.0.tar.gz",',
        '    ],',
        '    sha256 = '
        '"105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",',
        '    strip_prefix = "six-1.10.0",',
        '    build_file = "//third_party:six.BUILD",',
        ')',
    ]
    build_file = [
        'py_library(',
        '  name = "six",',
        '  srcs = ["six.py"],',
        ')',
    ]
    self.ScratchFile('WORKSPACE', rule_definition)
    self.ScratchFile('BUILD')
    self.ScratchFile('third_party/BUILD')
    self.ScratchFile('third_party/six.BUILD', build_file)

    exit_code, _, stderr = self.RunBazel(['build', '@six_archive//...'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))

    # Test specifying build_file as path
    # TODO(pcloudy):
    # Remove this after specifying build_file as path is no longer supported.
    rule_definition[-2] = 'build_file = "third_party/six.BUILD"'
    self.ScratchFile('WORKSPACE', rule_definition)
    exit_code, _, stderr = self.RunBazel(['build', '@six_archive//...'])
    self.assertEqual(exit_code, 0, os.linesep.join(stderr))

    # Test Repository reloading after build file changes
    self.ScratchFile('third_party/six.BUILD', build_file + ['foobar'])
    exit_code, _, stderr = self.RunBazel(['build', '@six_archive//...'])
    self.assertEqual(exit_code, 1, os.linesep.join(stderr))
    self.assertIn('name \'foobar\' is not defined.', os.linesep.join(stderr))


if __name__ == '__main__':
  unittest.main()
