# pylint: disable=g-backslash-continuation
# Copyright 2021 The Bazel Authors. All rights reserved.
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
import tempfile
import unittest

from src.test.py.bazel import test_base
from src.test.py.bazel.bzlmod.test_utils import BazelRegistry


class BazelModuleTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.registries_work_dir = tempfile.mkdtemp(dir=self._test_cwd)
    self.main_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'main'))
    self.main_registry.createCcModule('A', '1.0') \
        .createCcModule('A', '1.1') \
        .createCcModule('B', '1.0', {'A': '1.0'}, {'A': 'com_foo_bar_a'}) \
        .createCcModule('B', '1.1', {'A': '1.1'}) \
        .createCcModule('C', '1.1', {'A': '1.1', 'B': '1.1'})
    self.ScratchFile(
        '.bazelrc',
        [
            # In ipv6 only network, this has to be enabled.
            # 'startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true',
            'build --experimental_enable_bzlmod',
            'build --registry=' + self.main_registry.getURL(),
            # We need to have BCR here to make sure built-in modules like
            # bazel_tools can work.
            'build --registry=https://bcr.bazel.build',
            'build --verbose_failures',
        ])
    self.ScratchFile('WORKSPACE')
    # The existence of WORKSPACE.bzlmod prevents WORKSPACE prefixes or suffixes
    # from being used; this allows us to test built-in modules actually work
    self.ScratchFile('WORKSPACE.bzlmod')

  def writeMainProjectFiles(self):
    self.ScratchFile('A.patch', [
        '--- a/a.cc',
        '+++ b/a.cc',
        '@@ -1,6 +1,6 @@',
        ' #include <stdio.h>',
        ' #include "a.h"',
        ' void hello_a(const std::string& caller) {',
        '-    std::string lib_name = "A@1.0";',
        '+    std::string lib_name = "A@1.0 (locally patched)";',
        '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
        ' }',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = [',
        '    "@A//:lib_a",',
        '    "@B//:lib_b",',
        '  ],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "a.h"',
        '#include "b.h"',
        'int main() {',
        '    hello_a("main function");',
        '    hello_b("main function");',
        '}',
    ])

  def testSimple(self):
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.0")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@A//:lib_a"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "a.h"',
        'int main() {',
        '    hello_a("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0', stdout)

  def testSimpleTransitive(self):
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "B", version = "1.0")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@B//:lib_b"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "b.h"',
        'int main() {',
        '    hello_b("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => B@1.0', stdout)
    self.assertIn('B@1.0 => A@1.0', stdout)

  def testSimpleDiamond(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "A", version = "1.1")',
            # B1.0 has to depend on A1.1 after MVS.
            'bazel_dep(name = "B", version = "1.0")',
        ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.1', stdout)
    self.assertIn('main function => B@1.0', stdout)
    self.assertIn('B@1.0 => A@1.1', stdout)

  def testSingleVersionOverrideWithPatch(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "A", version = "1.1")',
            'bazel_dep(name = "B", version = "1.1")',
            # Both main and B1.1 has to depend on the locally patched A1.0.
            'single_version_override(',
            '  module_name = "A",',
            '  version = "1.0",',
            '  patches = ["//:A.patch"],',
            '  patch_strip = 1,',
            ')',
        ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0 (locally patched)', stdout)
    self.assertIn('main function => B@1.1', stdout)
    self.assertIn('B@1.1 => A@1.0 (locally patched)', stdout)

  def testRegistryOverride(self):
    self.writeMainProjectFiles()
    another_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'another'),
        ' from another registry')
    another_registry.createCcModule('A', '1.0')
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.0")',
        'bazel_dep(name = "B", version = "1.0")',
        'single_version_override(',
        '  module_name = "A",',
        '  registry = "%s",' % another_registry.getURL(),
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0 from another registry', stdout)
    self.assertIn('main function => B@1.0', stdout)
    self.assertIn('B@1.0 => A@1.0 from another registry', stdout)

  def testArchiveOverride(self):
    self.writeMainProjectFiles()
    archive_a_1_0 = self.main_registry.archives.joinpath('A.1.0.zip')
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.1")',
        'bazel_dep(name = "B", version = "1.1")',
        'archive_override(',
        '  module_name = "A",',
        '  urls = ["%s"],' % archive_a_1_0.as_uri(),
        '  patches = ["//:A.patch"],',
        '  patch_strip = 1,',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0 (locally patched)', stdout)
    self.assertIn('main function => B@1.1', stdout)
    self.assertIn('B@1.1 => A@1.0 (locally patched)', stdout)

  def testGitOverride(self):
    self.writeMainProjectFiles()

    src_a_1_0 = self.main_registry.projects.joinpath('A', '1.0')
    self.RunProgram(['git', 'init'], cwd=src_a_1_0, allow_failure=False)
    self.RunProgram(['git', 'config', 'user.name', 'tester'],
                    cwd=src_a_1_0,
                    allow_failure=False)
    self.RunProgram(['git', 'config', 'user.email', 'tester@foo.com'],
                    cwd=src_a_1_0,
                    allow_failure=False)
    self.RunProgram(['git', 'add', './'], cwd=src_a_1_0, allow_failure=False)
    self.RunProgram(['git', 'commit', '-m', 'Initial commit.'],
                    cwd=src_a_1_0,
                    allow_failure=False)
    _, stdout, _ = self.RunProgram(['git', 'rev-parse', 'HEAD'],
                                   cwd=src_a_1_0,
                                   allow_failure=False)
    commit = stdout[0].strip()

    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.1")',
        'bazel_dep(name = "B", version = "1.1")',
        'git_override(',
        '  module_name = "A",',
        '  remote = "%s",' % src_a_1_0.as_uri(),
        '  commit = "%s",' % commit,
        '  patches = ["//:A.patch"],',
        '  patch_strip = 1,',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0 (locally patched)', stdout)
    self.assertIn('main function => B@1.1', stdout)
    self.assertIn('B@1.1 => A@1.0 (locally patched)', stdout)

  def testLocalPathOverride(self):
    src_a_1_0 = self.main_registry.projects.joinpath('A', '1.0')
    self.writeMainProjectFiles()
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.1")',
        'bazel_dep(name = "B", version = "1.1")',
        'local_path_override(',
        '  module_name = "A",',
        '  path = "%s",' % str(src_a_1_0.resolve()).replace('\\', '/'),
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0', stdout)
    self.assertIn('main function => B@1.1', stdout)
    self.assertIn('B@1.1 => A@1.0', stdout)

  def testRemotePatchForBazelDep(self):
    patch_file = self.ScratchFile('A.patch', [
        '--- a/a.cc',
        '+++ b/a.cc',
        '@@ -1,6 +1,6 @@',
        ' #include <stdio.h>',
        ' #include "a.h"',
        ' void hello_a(const std::string& caller) {',
        '-    std::string lib_name = "A@1.1-1";',
        '+    std::string lib_name = "A@1.1-1 (remotely patched)";',
        '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
        ' }',
    ])
    self.main_registry.createCcModule(
        'A', '1.1-1', patches=[patch_file], patch_strip=1)
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.1-1")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@A//:lib_a"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "a.h"',
        'int main() {',
        '    hello_a("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.1-1 (remotely patched)', stdout)

  def testRepoNameForBazelDep(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "A", version = "1.0", repo_name = "my_repo_a_name")',
            # B should still be able to access A as com_foo_bar_a
            'bazel_dep(name = "B", version = "1.0")',
        ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = [',
        '    "@my_repo_a_name//:lib_a",',
        '    "@B//:lib_b",',
        '  ],',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => A@1.0', stdout)
    self.assertIn('main function => B@1.0', stdout)
    self.assertIn('B@1.0 => A@1.0', stdout)

  def testCheckDirectDependencies(self):
    self.writeMainProjectFiles()
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "A", version = "1.0")',
        'bazel_dep(name = "B", version = "1.0")',
        'bazel_dep(name = "C", version = "1.1")',
    ])
    _, stdout, stderr = self.RunBazel(
        ['run', '//:main', '--check_direct_dependencies=warning'],
        allow_failure=False)
    self.assertIn(
        'WARNING: For repository \'A\', the root module requires module version A@1.0, but got A@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn(
        'WARNING: For repository \'B\', the root module requires module version B@1.0, but got B@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn('main function => A@1.1', stdout)
    self.assertIn('main function => B@1.1', stdout)
    self.assertIn('B@1.1 => A@1.1', stdout)

    exit_code, _, stderr = self.RunBazel(
        ['run', '//:main', '--check_direct_dependencies=error'],
        allow_failure=True)
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        'ERROR: For repository \'A\', the root module requires module version A@1.0, but got A@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn(
        'ERROR: For repository \'B\', the root module requires module version B@1.0, but got B@1.1 in the resolved dependency graph.',
        stderr)

  def testRepositoryRuleErrorInModuleExtensionFailsTheBuild(self):
    self.ScratchFile('MODULE.bazel', [
        'module_ext = use_extension("//pkg:extension.bzl", "module_ext")',
        'use_repo(module_ext, "foo")',
    ])
    self.ScratchFile('pkg/BUILD.bazel')
    self.ScratchFile('pkg/rules.bzl', [
        'def _repo_rule_impl(ctx):',
        '    ctx.file("WORKSPACE")',
        'repo_rule = repository_rule(implementation = _repo_rule_impl)',
    ])
    self.ScratchFile('pkg/extension.bzl', [
        'load(":rules.bzl", "repo_rule")',
        'def _module_ext_impl(ctx):',
        '    repo_rule(name = "foo", invalid_attr = "value")',
        'module_ext = module_extension(implementation = _module_ext_impl)',
    ])
    exit_code, _, stderr = self.RunBazel(['run', '@foo//...'],
                                         allow_failure=True)
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        "ERROR: <builtin>: //pkg:@.module_ext.foo: no such attribute 'invalid_attr' in 'repo_rule' rule",
        stderr)
    self.assertTrue(
        any([
            '/pkg/extension.bzl", line 3, column 14, in _module_ext_impl'
            in line for line in stderr
        ]))


if __name__ == '__main__':
  unittest.main()
