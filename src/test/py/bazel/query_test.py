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

  def testQueryFilesUsedByRepositoryRules(self):
    self.ScratchFile('WORKSPACE')
    self._AssertQueryOutputContains(
        "kind('source file', deps(//external:*))",
        '@bazel_tools//tools/genrule:genrule-setup.sh',
    )

  def testBuildFilesForExternalRepos_Simple(self):
    self.ScratchFile('WORKSPACE', [
        'load("//:deps.bzl", "repos")',
        'repos()',
    ])
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('deps.bzl', [
        'def repos():',
        '    native.new_local_repository(',
        '        name = "io_bazel_rules_go",',
        '        path = ".",',
        """        build_file_content = "exports_files(glob(['*.go']))",""",
        '    )',
    ])
    self._AssertQueryOutputContains('buildfiles(//external:io_bazel_rules_go)',
                                    '//external:WORKSPACE', '//:deps.bzl',
                                    '//:BUILD.bazel')

  def testBuildFilesForExternalRepos_IndirectLoads(self):
    self.ScratchFile('WORKSPACE', [
        'load("//:deps.bzl", "repos")',
        'repos()',
    ])
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('deps.bzl', [
        'load("//:private_deps.bzl", "other_repos")',
        'def repos():',
        '    native.new_local_repository(',
        '        name = "io_bazel_rules_go",',
        '        path = ".",',
        """        build_file_content = "exports_files(glob(['*.go']))",""",
        '    )',
        '    other_repos()',
        '',
    ])
    self.ScratchFile('private_deps.bzl', [
        'def other_repos():',
        '    native.new_local_repository(',
        '        name = "io_bazel_rules_python",',
        '        path = ".",',
        """        build_file_content = "exports_files(glob(['*.py']))",""",
        '    )',
    ])

    self._AssertQueryOutputContains(
        'buildfiles(//external:io_bazel_rules_python)', '//external:WORKSPACE',
        '//:deps.bzl', '//:private_deps.bzl', '//:BUILD.bazel')

  def testBuildFilesForExternalRepos_NoDuplicates(self):
    self.ScratchFile('WORKSPACE', [
        'load("//:deps.bzl", "repos")',
        'repos()',
    ])
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('deps.bzl', [
        'def repos():',
        '    native.new_local_repository(',
        '        name = "io_bazel_rules_go",',
        '        path = ".",',
        """        build_file_content = "exports_files(glob(['*.go']))",""",
        '    )',
        '    other_repos()',
        '',
        'def other_repos():',
        '    native.new_local_repository(',
        '        name = "io_bazel_rules_python",',
        '        path = ".",',
        """        build_file_content = "exports_files(glob(['*.py']))",""",
        '    )',
    ])

    _, stdout, _ = self.RunBazel(
        ['query', 'buildfiles(//external:io_bazel_rules_python)']
    )
    result = set()
    for item in stdout:
      if not item:
        continue
      self.assertNotIn(item, result)
      result.add(item)

  def _AssertQueryOutput(self, query_expr, *expected_results):
    _, stdout, _ = self.RunBazel(['query', query_expr])

    stdout = sorted(x for x in stdout if x)
    self.assertEqual(len(stdout), len(expected_results))
    self.assertListEqual(stdout, sorted(expected_results))

  def _AssertQueryOutputContains(self, query_expr, *expected_content):
    _, stdout, _ = self.RunBazel(['query', query_expr])

    stdout = {x for x in stdout if x}
    for item in expected_content:
      self.assertIn(item, stdout)


if __name__ == '__main__':
  unittest.main()
