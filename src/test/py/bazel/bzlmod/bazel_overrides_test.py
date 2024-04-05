# pylint: disable=g-backslash-continuation
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# pylint: disable=g-long-ternary

import os
import tempfile
from absl.testing import absltest
from src.test.py.bazel import test_base
from src.test.py.bazel.bzlmod.test_utils import BazelRegistry


class BazelOverridesTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.registries_work_dir = tempfile.mkdtemp(dir=self._test_cwd)
    self.main_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'main')
    )
    self.main_registry.start()
    self.main_registry.createCcModule('aaa', '1.0').createCcModule(
        'aaa', '1.1'
    ).createCcModule('bbb', '1.0', {'aaa': '1.0'}).createCcModule(
        'bbb', '1.1', {'aaa': '1.1'}
    ).createCcModule(
        'ccc', '1.1', {'aaa': '1.1', 'bbb': '1.1'}
    )
    self.ScratchFile(
        '.bazelrc',
        [
            # In ipv6 only network, this has to be enabled.
            # 'startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true',
            'build --noenable_workspace',
            'build --registry=' + self.main_registry.getURL(),
            # We need to have BCR here to make sure built-in modules like
            # bazel_tools can work.
            'build --registry=https://bcr.bazel.build',
            'build --verbose_failures',
            # Set an explicit Java language version
            'build --java_language_version=8',
            'build --tool_java_language_version=8',
            'build --lockfile_mode=update',
        ],
    )

  def tearDown(self):
      self.main_registry.stop()
      test_base.TestBase.tearDown(self)

  def writeMainProjectFiles(self):
    self.ScratchFile(
        'aaa.patch',
        [
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
        ],
    )
    self.ScratchFile(
        'BUILD',
        [
            'cc_binary(',
            '  name = "main",',
            '  srcs = ["main.cc"],',
            '  deps = [',
            '    "@aaa//:lib_aaa",',
            '    "@bbb//:lib_bbb",',
            '  ],',
            ')',
        ],
    )
    self.ScratchFile(
        'main.cc',
        [
            '#include "aaa.h"',
            '#include "bbb.h"',
            'int main() {',
            '    hello_aaa("main function");',
            '    hello_bbb("main function");',
            '}',
        ],
    )

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
        ],
    )
    _, stdout, _ = self.RunBazel(['run', '//:main'])
    self.assertIn('main function => aaa@1.0 (locally patched)', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched)', stdout)

  def testRegistryOverride(self):
    self.writeMainProjectFiles()
    another_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'another'),
        ' from another registry',
    )
    another_registry.start()
    try:
      another_registry.createCcModule('aaa', '1.0')
      self.ScratchFile(
          'MODULE.bazel',
          [
              'bazel_dep(name = "aaa", version = "1.0")',
              'bazel_dep(name = "bbb", version = "1.0")',
              'single_version_override(',
              '  module_name = "aaa",',
              '  registry = "%s",' % another_registry.getURL(),
              ')',
          ],
      )
      _, stdout, _ = self.RunBazel(['run', '//:main'])
      self.assertIn('main function => aaa@1.0 from another registry', stdout)
      self.assertIn('main function => bbb@1.0', stdout)
      self.assertIn('bbb@1.0 => aaa@1.0 from another registry', stdout)
    finally:
      another_registry.stop()

  def testArchiveOverride(self):
    self.writeMainProjectFiles()
    archive_aaa_1_0 = self.main_registry.archives.joinpath('aaa.1.0.zip')
    self.ScratchFile(
        'aaa2.patch',
        [
            '--- a/aaa.cc',
            '+++ b/aaa.cc',
            '@@ -1,6 +1,6 @@',
            ' #include <stdio.h>',
            ' #include "aaa.h"',
            ' void hello_aaa(const std::string& caller) {',
            '-    std::string lib_name = "aaa@1.0 (locally patched)";',
            '+    std::string lib_name = "aaa@1.0 (locally patched again)";',
            '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
            ' }',
        ],
    )
    self.ScratchFile(
        'aaa3.patch',
        [
            '--- a/aaa.cc',
            '+++ b/aaa.cc',
            '@@ -1,6 +1,6 @@',
            ' #include <stdio.h>',
            ' #include "aaa.h"',
            ' void hello_aaa(const std::string& caller) {',
            '-    std::string lib_name = "aaa@1.0 (locally patched again)";',
            (
                '+    std::string lib_name = "aaa@1.0 (locally patched again'
                ' and again)";'
            ),
            '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
            ' }',
        ],
    )
    self.ScratchFile(
        'aaa4.patch',
        [
            '--- a/aaa.cc',
            '+++ b/aaa.cc',
            '@@ -1,6 +1,6 @@',
            ' #include <stdio.h>',
            ' #include "aaa.h"',
            ' void hello_aaa(const std::string& caller) {',
            (
                '-    std::string lib_name = "aaa@1.0 (locally patched again'
                ' and again)";'
            ),
            (
                '+    std::string lib_name = "aaa@1.0 (locally patched all over'
                ' again)";'
            ),
            '     printf("%s => %s\\n", caller.c_str(), lib_name.c_str());',
            ' }',
        ],
    )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'module(name = "main", repo_name = "my_main")',
            'bazel_dep(name = "aaa", version = "1.1")',
            'bazel_dep(name = "bbb", version = "1.1")',
            'archive_override(',
            '  module_name = "aaa",',
            '  urls = ["%s"],' % archive_aaa_1_0.as_uri(),
            '  patches = [',
            '    "//:aaa.patch",',
            '    "@//:aaa2.patch",',
            '    "@my_main//:aaa3.patch",',
            '    ":aaa4.patch",',
            '  ],',
            '  patch_strip = 1,',
            ')',
        ],
    )
    _, stdout, _ = self.RunBazel(['run', '//:main'])
    self.assertIn(
        'main function => aaa@1.0 (locally patched all over again)', stdout
    )
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched all over again)', stdout)

  def testGitOverride(self):
    self.writeMainProjectFiles()
    src_aaa_1_0 = self.main_registry.projects.joinpath('aaa', '1.0')
    self.RunProgram(['git', 'init'], cwd=src_aaa_1_0)
    self.RunProgram(
        ['git', 'config', 'user.name', 'tester'],
        cwd=src_aaa_1_0,
    )
    self.RunProgram(
        ['git', 'config', 'user.email', 'tester@foo.com'],
        cwd=src_aaa_1_0,
    )
    self.RunProgram(['git', 'add', './'], cwd=src_aaa_1_0)
    self.RunProgram(
        ['git', 'commit', '-m', 'Initial commit.'],
        cwd=src_aaa_1_0,
    )
    _, stdout, _ = self.RunProgram(
        ['git', 'rev-parse', 'HEAD'], cwd=src_aaa_1_0
    )
    commit = stdout[0].strip()

    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.1")',
            'bazel_dep(name = "bbb", version = "1.1")',
            'git_override(',
            '  module_name = "aaa",',
            '  remote = "%s",' % src_aaa_1_0.as_uri(),
            '  commit = "%s",' % commit,
            '  patches = ["//:aaa.patch"],',
            '  patch_strip = 1,',
            ')',
        ],
    )
    _, stdout, _ = self.RunBazel(['run', '//:main'])
    self.assertIn('main function => aaa@1.0 (locally patched)', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0 (locally patched)', stdout)

  def testLocalPathOverride(self):
    src_aaa_1_0 = self.main_registry.projects.joinpath('aaa', '1.0')
    self.writeMainProjectFiles()
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.1")',
            'bazel_dep(name = "bbb", version = "1.1")',
            'local_path_override(',
            '  module_name = "aaa",',
            '  path = "%s",' % str(src_aaa_1_0.resolve()).replace('\\', '/'),
            ')',
        ],
    )
    _, stdout, _ = self.RunBazel(['run', '//:main'])
    self.assertIn('main function => aaa@1.0', stdout)
    self.assertIn('main function => bbb@1.1', stdout)
    self.assertIn('bbb@1.1 => aaa@1.0', stdout)

  def testCmdAbsoluteModuleOverride(self):
    # test commandline_overrides takes precedence over local_path_override
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "ss", version = "1.0")',
            'local_path_override(',
            '  module_name = "ss",',
            '  path = "%s",' % self.Path('aa'),
            ')',
        ],
    )
    self.ScratchFile('BUILD')

    self.ScratchFile(
        'aa/MODULE.bazel',
        [
            "module(name='ss')",
        ],
    )
    self.ScratchFile(
        'aa/BUILD',
        [
            'filegroup(name = "never_ever")',
        ],
    )

    self.ScratchFile(
        'bb/MODULE.bazel',
        [
            "module(name='ss')",
        ],
    )
    self.ScratchFile(
        'bb/BUILD',
        [
            'filegroup(name = "choose_me")',
        ],
    )

    _, _, stderr = self.RunBazel(
        ['build', '@ss//:all', '--override_module', 'ss=' + self.Path('bb')]
    )
    # module file override should be ignored, and bb directory should be used
    self.assertIn(
        'Target @@ss~//:choose_me up-to-date (nothing to build)', stderr
    )

  def testCmdRelativeModuleOverride(self):
    self.ScratchFile(
        'aa/MODULE.bazel',
        [
            'bazel_dep(name = "ss", version = "1.0")',
        ],
    )
    self.ScratchFile('aa/BUILD')

    self.ScratchFile('aa/cc/BUILD')

    self.ScratchFile(
        'bb/MODULE.bazel',
        [
            'module(name="ss")',
        ],
    )
    self.ScratchFile(
        'bb/BUILD',
        [
            'filegroup(name = "choose_me")',
        ],
    )

    _, _, stderr = self.RunBazel(
        [
            'build',
            '@ss//:all',
            '--override_module',
            'ss=../../bb',
            '--enable_bzlmod',
        ],
        cwd=self.Path('aa/cc'),
    )
    self.assertIn(
        'Target @@ss~//:choose_me up-to-date (nothing to build)', stderr
    )

  def testCmdWorkspaceRelativeModuleOverride(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "ss", version = "1.0")',
        ],
    )
    self.ScratchFile('BUILD')
    self.ScratchFile('aa/BUILD')
    self.ScratchFile(
        'bb/MODULE.bazel',
        [
            'module(name="ss")',
        ],
    )
    self.ScratchFile(
        'bb/BUILD',
        [
            'filegroup(name = "choose_me")',
        ],
    )

    _, _, stderr = self.RunBazel(
        [
            'build',
            '@ss//:all',
            '--override_module',
            'ss=%workspace%/bb',
        ],
        cwd=self.Path('aa'),
    )
    self.assertIn(
        'Target @@ss~//:choose_me up-to-date (nothing to build)', stderr
    )


if __name__ == '__main__':
  absltest.main()
