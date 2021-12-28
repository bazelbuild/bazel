# pylint: disable=g-bad-file-header
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


class QueryTest(test_base.TestBase):

  def testSimpleQuery(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'exports_files(["exported.txt"])',
        'filegroup(name = "top-rule", srcs = [":dep-rule"])',
        'filegroup(name = "dep-rule", srcs = ["src.txt"])',
    ])
    self.ScratchFile('foo/src.txt')
    self.ScratchFile('foo/exported.txt')
    self.ScratchFile('foo/non-exported.txt')
    self._AssertQueryOutput('//foo:top-rule', '//foo:top-rule')
    self._AssertQueryOutput('//foo:*', '//foo:top-rule', '//foo:dep-rule',
                            '//foo:src.txt', '//foo:exported.txt',
                            '//foo:BUILD')
    self._AssertQueryOutput('deps(//foo:top-rule)', '//foo:top-rule',
                            '//foo:dep-rule', '//foo:src.txt')
    self._AssertQueryOutput('deps(//foo:top-rule, 1)', '//foo:top-rule',
                            '//foo:dep-rule')

  def testBuildFilesForExternalRepos(self):
    self.ScratchFile('WORKSPACE', [
      'load("//:http_archives.bzl", "http_archives")',
      'http_archives()',
    ])
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('http_archives.bzl', [
      'load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")',
      'def http_archives():',
      '    http_archive(',
      '        name = "io_bazel_rules_go",',
      '        sha256 = "2b1641428dff9018f9e85c0384f03ec6c10660d935b750e3fa1492a281a53b0f",',
      '        urls = [',
      '            "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.29.0/rules_go-v0.29.0.zip",',
      '            "https://github.com/bazelbuild/rules_go/releases/download/v0.29.0/rules_go-v0.29.0.zip",',
      '        ],',
      '    )',
    ])
    self._AssertQueryOutput('buildfiles(//external:io_bazel_rules_go)',
                            '//:WORKSPACE', '//:http_archives.bzl', '//:BUILD.bazel')

  def _AssertQueryOutput(self, query_expr, *expected_results):
    exit_code, stdout, stderr = self.RunBazel(['query', query_expr])
    self.AssertExitCode(exit_code, 0, stderr)

    stdout = sorted(x for x in stdout if x)
    self.assertEqual(len(stdout), len(expected_results))
    self.assertListEqual(stdout, sorted(expected_results))


if __name__ == '__main__':
  unittest.main()
