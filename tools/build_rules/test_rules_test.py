# Copyright 2018 The Bazel Authors. All rights reserved.
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

import unittest

from src.test.py.bazel import test_base


class TestRulesTest(test_base.TestBase):

  def _FailWithOutput(self, output):
    self.fail('FAIL:\n | %s\n---' % '\n | '.join(output))

  def _AssertPasses(self, target):
    exit_code, stdout, stderr = self.RunBazel(
        ['test', target, '--test_output=errors'])
    if exit_code != 0:
      self._FailWithOutput(stdout + stderr)

  def _AssertFails(self, target):
    exit_code, stdout, stderr = self.RunBazel(['test', target])
    if exit_code == 0:
      self._FailWithOutput(stdout + stderr)

  def testContent(self):
    self.ScratchFile('WORKSPACE')
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_rules/test_rules.bzl'),
        'foo/test_rules.bzl')
    self.ScratchFile('foo/tested_file.txt',
                     ['The quick brown', 'fox jumps over', 'the lazy dog.'])
    self.ScratchFile('foo/BUILD', [
        'load(":test_rules.bzl", "file_test")',
        '',
        'file_test(',
        '    name = "pos",',
        '    content = "The quick brown\\nfox jumps over\\nthe lazy dog.\\n",',
        '    file = "tested_file.txt",',
        ')',
        '',
        'file_test(',
        '    name = "neg",',
        '    content = "quick",',
        '    file = "tested_file.txt",',
        ')',
    ])
    self._AssertPasses('//foo:pos')
    self._AssertFails('//foo:neg')

  def testRegexpWithoutMatches(self):
    self.ScratchFile('WORKSPACE')
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_rules/test_rules.bzl'),
        'foo/test_rules.bzl')
    self.ScratchFile('foo/tested_file.txt',
                     ['The quick brown', 'fox jumps over', 'the lazy dog.'])
    self.ScratchFile('foo/BUILD', [
        'load(":test_rules.bzl", "file_test")',
        '',
        'file_test(',
        '    name = "pos",',
        '    file = "tested_file.txt",',
        '    regexp = "o[vwx]",',
        ')',
        '',
        'file_test(',
        '    name = "neg",',
        '    file = "tested_file.txt",',
        '    regexp = "o[abc]",',
        ')',
    ])
    self._AssertPasses('//foo:pos')
    self._AssertFails('//foo:neg')

  def testRegexpWithMatches(self):
    self.ScratchFile('WORKSPACE')
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_rules/test_rules.bzl'),
        'foo/test_rules.bzl')
    self.ScratchFile('foo/tested_file.txt',
                     ['The quick brown', 'fox jumps over', 'the lazy dog.'])
    self.ScratchFile(
        'foo/BUILD',
        [
            'load(":test_rules.bzl", "file_test")',
            '',
            'file_test(',
            '    name = "pos",',
            '    file = "tested_file.txt",',
            # grep -c returns the number of matching lines, not the number of
            # matches
            '    matches = 2,',
            '    regexp = "o[vwx]",',
            ')',
            '',
            'file_test(',
            '    name = "neg",',
            '    file = "tested_file.txt",',
            '    matches = 3,',
            '    regexp = "o[vwx]",',
            ')',
        ])
    self._AssertPasses('//foo:pos')
    self._AssertFails('//foo:neg')

  def testBadArgs(self):
    self.ScratchFile('WORKSPACE')
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_rules/test_rules.bzl'),
        'foo/test_rules.bzl')
    self.ScratchFile('foo/tested_file.txt',
                     ['The quick brown', 'fox jumps over', 'the lazy dog.'])
    self.ScratchFile('foo/BUILD', [
        'load(":test_rules.bzl", "file_test")',
        '',
        'file_test(',
        '    name = "neither_content_nor_regex",',
        '    file = "tested_file.txt",',
        ')',
        '',
        'file_test(',
        '    name = "both_content_and_regex",',
        '    file = "tested_file.txt",',
        '    content = "x",',
        '    regexp = "x",',
        ')',
        '',
        'file_test(',
        '    name = "content_with_matches",',
        '    file = "tested_file.txt",',
        '    content = "hello",',
        '    matches = 1,',
        ')',
    ])
    self._AssertFails('//foo:neither_content_nor_regex')
    self._AssertFails('//foo:both_content_and_regex')
    self._AssertFails('//foo:content_with_matches')


if __name__ == '__main__':
  unittest.main()
