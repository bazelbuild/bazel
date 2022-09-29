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
import pathlib
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
    self.main_registry.createCcModule('aaa', '1.0') \
        .createCcModule('aaa', '1.1') \
        .createCcModule('bbb', '1.0', {'aaa': '1.0'}, {'aaa': 'com_foo_aaa'}) \
        .createCcModule('bbb', '1.1', {'aaa': '1.1'}) \
        .createCcModule('ccc', '1.1', {'aaa': '1.1', 'bbb': '1.1'})
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
    self.ScratchFile('aaa.patch', [
        '--- a/aaa.cc',
        '+++ b/aaa.cc',
        '@@ -1,6 +1,6 @@',
        ' #include <stdio.h>',
        ' #include "aaa.h"',
        ' void hello_aaa(const std::string& caller) {',
        '-    std::string lib_name = "aaa@1.0";',
        '+    std::string lib_name = "aaa@1.0 (locally patched)";',
        '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
        ' }',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = [',
        '    "@aaa//:lib_aaa",',
        '    "@bbb//:lib_bbb",',
        '  ],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "aaa.h"',
        '#include "bbb.h"',
        'int main() {',
        '    hello_aaa("main function");',
        '    hello_bbb("main function");',
        '}',
    ])

  def testSimple(self):
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.0")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@aaa//:lib_aaa"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "aaa.h"',
        'int main() {',
        '    hello_aaa("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0', stdout)

  def testSimpleTransitive(self):
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "bbb", version = "1.0")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@bbb//:lib_bbb"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "bbb.h"',
        'int main() {',
        '    hello_bbb("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => bbb@1.0', stdout)
    self.assertIn('bbb@1.0 => aaa@1.0', stdout)

  def testSimpleDiamond(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.1")',
            # bbb@1.0 has to depend on aaa@1.1 after MVS.
            'bazel_dep(name = "bbb", version = "1.0")',
        ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.1', stdout)
    self.assertIn('main function => bbb@1.0', stdout)
    self.assertIn('bbb@1.0 => aaa@1.1', stdout)

  def testSingleVersionOverrideWithPatch(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.1")',
            'bazel_dep(name = "bbb", version = "1.1")',
            # Both main and bbb@1.1 has to depend on the locally patched aaa@1.0
            'single_version_override(',
            '  module_name = "aaa",',
            '  version = "1.0",',
            '  patches = ["//:aaa.patch"],',
            '  patch_strip = 1,',
            ')',
        ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0 (locally patched)', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched)', stdout)

  def testRegistryOverride(self):
    self.writeMainProjectFiles()
    another_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'another'),
        ' from another registry')
    another_registry.createCcModule('aaa', '1.0')
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.0")',
        'bazel_dep(name = "bbb", version = "1.0")',
        'single_version_override(',
        '  module_name = "aaa",',
        '  registry = "%s",' % another_registry.getURL(),
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0 from another registry', stdout)
    self.assertIn('main function => bbb@1.0', stdout)
    self.assertIn('bbb@1.0 => aaa@1.0 from another registry', stdout)

  def testArchiveOverride(self):
    self.writeMainProjectFiles()
    archive_aaa_1_0 = self.main_registry.archives.joinpath('aaa.1.0.zip')
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.1")',
        'bazel_dep(name = "bbb", version = "1.1")',
        'archive_override(',
        '  module_name = "aaa",',
        '  urls = ["%s"],' % archive_aaa_1_0.as_uri(),
        '  patches = ["//:aaa.patch"],',
        '  patch_strip = 1,',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0 (locally patched)', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched)', stdout)

  def testGitOverride(self):
    self.writeMainProjectFiles()

    src_aaa_1_0 = self.main_registry.projects.joinpath('aaa', '1.0')
    self.RunProgram(['git', 'init'], cwd=src_aaa_1_0, allow_failure=False)
    self.RunProgram(['git', 'config', 'user.name', 'tester'],
                    cwd=src_aaa_1_0,
                    allow_failure=False)
    self.RunProgram(['git', 'config', 'user.email', 'tester@foo.com'],
                    cwd=src_aaa_1_0,
                    allow_failure=False)
    self.RunProgram(['git', 'add', './'], cwd=src_aaa_1_0, allow_failure=False)
    self.RunProgram(['git', 'commit', '-m', 'Initial commit.'],
                    cwd=src_aaa_1_0,
                    allow_failure=False)
    _, stdout, _ = self.RunProgram(['git', 'rev-parse', 'HEAD'],
                                   cwd=src_aaa_1_0,
                                   allow_failure=False)
    commit = stdout[0].strip()

    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.1")',
        'bazel_dep(name = "bbb", version = "1.1")',
        'git_override(',
        '  module_name = "aaa",',
        '  remote = "%s",' % src_aaa_1_0.as_uri(),
        '  commit = "%s",' % commit,
        '  patches = ["//:aaa.patch"],',
        '  patch_strip = 1,',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0 (locally patched)', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched)', stdout)

  def testLocalPathOverride(self):
    src_aaa_1_0 = self.main_registry.projects.joinpath('aaa', '1.0')
    self.writeMainProjectFiles()
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.1")',
        'bazel_dep(name = "bbb", version = "1.1")',
        'local_path_override(',
        '  module_name = "aaa",',
        '  path = "%s",' % str(src_aaa_1_0.resolve()).replace('\\', '/'),
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0', stdout)

  def testRemotePatchForBazelDep(self):
    patch_file = self.ScratchFile('aaa.patch', [
        '--- a/aaa.cc',
        '+++ b/aaa.cc',
        '@@ -1,6 +1,6 @@',
        ' #include <stdio.h>',
        ' #include "aaa.h"',
        ' void hello_aaa(const std::string& caller) {',
        '-    std::string lib_name = "aaa@1.1-1";',
        '+    std::string lib_name = "aaa@1.1-1 (remotely patched)";',
        '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
        ' }',
    ])
    self.main_registry.createCcModule(
        'aaa', '1.1-1', patches=[patch_file], patch_strip=1)
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.1-1")',
    ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = ["@aaa//:lib_aaa"],',
        ')',
    ])
    self.ScratchFile('main.cc', [
        '#include "aaa.h"',
        'int main() {',
        '    hello_aaa("main function");',
        '}',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.1-1 (remotely patched)', stdout)

  def testRepoNameForBazelDep(self):
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.0", repo_name = "my_repo_a_name")',
            # bbb should still be able to access aaa as com_foo_aaa
            'bazel_dep(name = "bbb", version = "1.0")',
        ])
    self.ScratchFile('BUILD', [
        'cc_binary(',
        '  name = "main",',
        '  srcs = ["main.cc"],',
        '  deps = [',
        '    "@my_repo_a_name//:lib_aaa",',
        '    "@bbb//:lib_bbb",',
        '  ],',
        ')',
    ])
    _, stdout, _ = self.RunBazel(['run', '//:main'], allow_failure=False)
    self.assertIn('main function => aaa@1.0', stdout)
    self.assertIn('main function => bbb@1.0', stdout)
    self.assertIn('bbb@1.0 => aaa@1.0', stdout)

  def testCheckDirectDependencies(self):
    self.writeMainProjectFiles()
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "aaa", version = "1.0")',
        'bazel_dep(name = "bbb", version = "1.0")',
        'bazel_dep(name = "ccc", version = "1.1")',
    ])
    _, stdout, stderr = self.RunBazel(
        ['run', '//:main', '--check_direct_dependencies=warning'],
        allow_failure=False)
    self.assertIn(
        'WARNING: For repository \'aaa\', the root module requires module version aaa@1.0, but got aaa@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn(
        'WARNING: For repository \'bbb\', the root module requires module version bbb@1.0, but got bbb@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn('main function => aaa@1.1', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.1', stdout)

    exit_code, _, stderr = self.RunBazel(
        ['run', '//:main', '--check_direct_dependencies=error'],
        allow_failure=True)
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        'ERROR: For repository \'aaa\', the root module requires module version aaa@1.0, but got aaa@1.1 in the resolved dependency graph.',
        stderr)
    self.assertIn(
        'ERROR: For repository \'bbb\', the root module requires module version bbb@1.0, but got bbb@1.1 in the resolved dependency graph.',
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
        "ERROR: <builtin>: //pkg:~module_ext~foo: no such attribute 'invalid_attr' in 'repo_rule' rule",
        stderr)
    self.assertTrue(
        any([
            '/pkg/extension.bzl", line 3, column 14, in _module_ext_impl'
            in line for line in stderr
        ]))

  def testCommandLineModuleOverride(self):
    self.ScratchFile('MODULE.bazel', [
        'bazel_dep(name = "ss", version = "1.0")',
        'local_path_override(',
        '  module_name = "ss",',
        '  path = "%s",' % self.Path('aa'),
        ')',
    ])
    self.ScratchFile('BUILD')
    self.ScratchFile('WORKSPACE')

    self.ScratchFile('aa/MODULE.bazel', [
        'module(name=\'ss\')',
    ])
    self.ScratchFile('aa/BUILD', [
        'filegroup(name = "never_ever")',
    ])
    self.ScratchFile('aa/WORKSPACE')

    self.ScratchFile('bb/MODULE.bazel', [
        'module(name=\'ss\')',
    ])
    self.ScratchFile('bb/BUILD', [
        'filegroup(name = "choose_me")',
    ])
    self.ScratchFile('bb/WORKSPACE')

    _, _, stderr = self.RunBazel([
        'build', '--experimental_enable_bzlmod', '@ss//:all',
        '--override_module', 'ss=' + self.Path('bb')
    ],
                                 allow_failure=False)
    # module file override should be ignored, and bb directory should be used
    self.assertIn(
        'Target @ss~override//:choose_me up-to-date (nothing to build)', stderr)

  def testDownload(self):
    data_path = self.ScratchFile('data.txt', ['some data'])
    data_url = pathlib.Path(data_path).resolve().as_uri()
    self.ScratchFile('MODULE.bazel', [
        'data_ext = use_extension("//:ext.bzl", "data_ext")',
        'use_repo(data_ext, "no_op")',
    ])
    self.ScratchFile('BUILD')
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('ext.bzl', [
        'def _no_op_impl(ctx):',
        '  ctx.file("WORKSPACE")',
        '  ctx.file("BUILD", "filegroup(name=\\"no_op\\")")',
        'no_op = repository_rule(_no_op_impl)',
        'def _data_ext_impl(ctx):',
        '  if not ctx.download(url="%s", output="data.txt").success:' %
        data_url,
        '    fail("download failed")',
        '  if ctx.read("data.txt").strip() != "some data":',
        '    fail("unexpected downloaded content: %s" % ctx.read("data.txt").strip())',
        '  no_op(name="no_op")',
        'data_ext = module_extension(_data_ext_impl)',
    ])
    self.RunBazel(['build', '@no_op//:no_op'], allow_failure=False)

if __name__ == '__main__':
  unittest.main()
