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


class BazelFetchTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.registries_work_dir = tempfile.mkdtemp(dir=self._test_cwd)
    self.main_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'main')
    )
    self.ScratchFile(
        '.bazelrc',
        [
            # In ipv6 only network, this has to be enabled.
            # 'startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true',
            'common --enable_bzlmod',
            'common --experimental_isolated_extension_usages',
            'common --registry=' + self.main_registry.getURL(),
            'common --verbose_failures',
            # Set an explicit Java language version
            'common --java_language_version=8',
            'common --tool_java_language_version=8',
            'common --lockfile_mode=update',
        ],
    )
    self.ScratchFile('WORKSPACE')
    # The existence of WORKSPACE.bzlmod prevents WORKSPACE prefixes or suffixes
    # from being used; this allows us to test built-in modules actually work
    self.ScratchFile('WORKSPACE.bzlmod')
    self.generatBuiltinModules()

  def generatBuiltinModules(self):
    self.ScratchFile('platforms_mock/BUILD')
    self.ScratchFile('platforms_mock/WORKSPACE')
    self.ScratchFile(
        'platforms_mock/MODULE.bazel', ['module(name="local_config_platform")']
    )

    self.ScratchFile('tools_mock/BUILD')
    self.ScratchFile('tools_mock/WORKSPACE')
    self.ScratchFile('tools_mock/MODULE.bazel', ['module(name="bazel_tools")'])
    self.ScratchFile('tools_mock/tools/build_defs/repo/BUILD')
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_defs/repo/http.bzl'),
        'tools_mock/tools/build_defs/repo/http.bzl',
    )
    self.CopyFile(
        self.Rlocation('io_bazel/tools/build_defs/repo/utils.bzl'),
        'tools_mock/tools/build_defs/repo/utils.bzl',
    )

  def testFetchAll(self):
    self.main_registry.createCcModule('aaa', '1.0').createCcModule(
        'bbb', '1.0', {'aaa': '1.0'}
    )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "bbb", version = "1.0")',
            'ext = use_extension("extension.bzl", "ext")',
            'use_repo(ext, "hello")',
            'local_path_override(module_name="bazel_tools", path="tools_mock")',
            'local_path_override(module_name="local_config_platform", ',
            'path="platforms_mock")',
        ],
    )
    self.ScratchFile('BUILD')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _ext_impl(ctx):',
            '    repo_rule(name="hello")',
            'ext = module_extension(implementation=_ext_impl)',
        ],
    )

    self.RunBazel(['fetch', '--all'])
    _, stdout, _ = self.RunBazel(['info', 'output_base'])
    repos_fetched = os.listdir(stdout[0] + '/external')
    self.assertIn('aaa~1.0', repos_fetched)
    self.assertIn('bbb~1.0', repos_fetched)
    self.assertIn('_main~ext~hello', repos_fetched)

  def testFetchConfig(self):
    self.main_registry.createCcModule('aaa', '1.0').createCcModule(
        'bbb', '1.0', {'aaa': '1.0'}
    )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "aaa", version = "1.0")',
            'ext = use_extension("extension.bzl", "ext")',
            'use_repo(ext, "notConfig")',
            'use_repo(ext, "IamConfig")',
            'local_path_override(module_name="bazel_tools", path="tools_mock")',
            'local_path_override(module_name="local_config_platform", ',
            'path="platforms_mock")',
        ],
    )
    self.ScratchFile('BUILD')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            'repo_rule2 = repository_rule(implementation=_repo_rule_impl, ',
            'configure=True)',
            '',
            'def _ext_impl(ctx):',
            '    repo_rule(name="notConfig")',
            '    repo_rule2(name="IamConfig")',
            'ext = module_extension(implementation=_ext_impl)',
        ],
    )

    self.RunBazel(['fetch', '--configure'])
    _, stdout, _ = self.RunBazel(['info', 'output_base'])
    repos_fetched = os.listdir(stdout[0] + '/external')
    self.assertNotIn('aaa~1.0', repos_fetched)
    self.assertNotIn('_main~ext~notConfig', repos_fetched)
    self.assertIn('_main~ext~IamConfig', repos_fetched)

  def testFetchFailsWithMultipleOptions(self):
    exit_code, _, stderr = self.RunBazel(
        ['fetch', '--all', '--configure'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 2, stderr)
    self.assertIn(
        'ERROR: Only one fetch option should be provided for fetch command.',
        stderr,
    )
    exit_code, _, stderr = self.RunBazel(
        ['fetch', '--all', '--repo=@hello'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 2, stderr)
    self.assertIn(
        'ERROR: Only one fetch option should be provided for fetch command.',
        stderr,
    )

  def testFetchRepo(self):
    self.main_registry.createCcModule('aaa', '1.0').createCcModule(
        'bbb', '1.0', {'aaa': '1.0'}
    ).createCcModule('ccc', '1.0')
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "bbb", version = "1.0")',
            'bazel_dep(name = "ccc", version = "1.0", repo_name = "my_repo")',
            'local_path_override(module_name="bazel_tools", path="tools_mock")',
            'local_path_override(module_name="local_config_platform", ',
            'path="platforms_mock")',
        ],
    )
    self.ScratchFile('BUILD')
    # Test canonical/apparent repo names & multiple repos
    self.RunBazel(['fetch', '--repo=@@bbb~1.0', '--repo=@my_repo'])
    _, stdout, _ = self.RunBazel(['info', 'output_base'])
    repos_fetched = os.listdir(stdout[0] + '/external')
    self.assertIn('bbb~1.0', repos_fetched)
    self.assertIn('ccc~1.0', repos_fetched)
    self.assertNotIn('aaa~1.0', repos_fetched)

  def testFetchInvalidRepo(self):
    # Invalid repo name (not canonical or apparent)
    exit_code, _, stderr = self.RunBazel(
        ['fetch', '--repo=hello'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 8, stderr)
    self.assertIn(
        'ERROR: Invalid repo name: The repo value has to be either apparent'
        " '@repo' or canonical '@@repo' repo name",
        stderr,
    )
    # Repo does not exist
    self.ScratchFile(
        'MODULE.bazel',
        [
            'local_path_override(module_name="bazel_tools", path="tools_mock")',
            'local_path_override(module_name="local_config_platform", ',
            'path="platforms_mock")',
        ],
    )
    exit_code, _, stderr = self.RunBazel(
        ['fetch', '--repo=@@nono', '--repo=@nana'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 8, stderr)
    self.assertIn(
        "ERROR: Fetching repos failed with errors: Repository '@nono' is not "
        "defined; No repository visible as '@nana' from main repository",
        stderr,
    )


if __name__ == '__main__':
  absltest.main()
