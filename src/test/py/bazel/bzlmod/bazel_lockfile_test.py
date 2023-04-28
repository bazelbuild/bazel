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
import unittest

from src.test.py.bazel import test_base
from src.test.py.bazel.bzlmod.test_utils import BazelRegistry
from src.test.py.bazel.bzlmod.test_utils import scratchFile


class BazelLockfileTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.registries_work_dir = tempfile.mkdtemp(dir=self._test_cwd)
    self.main_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'main')
    )
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
            'build --enable_bzlmod',
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
    self.ScratchFile('WORKSPACE')
    # The existence of WORKSPACE.bzlmod prevents WORKSPACE prefixes or suffixes
    # from being used; this allows us to test built-in modules actually work
    self.ScratchFile('WORKSPACE.bzlmod')

  def testChangeModuleInRegistryWithoutLockfile(self):
    # Add module 'sss' to the registry with dep on 'aaa'
    self.main_registry.createCcModule('sss', '1.3', {'aaa': '1.1'})
    # Create a project with deps on 'sss'
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "sss", version = "1.3")',
        ],
    )
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.RunBazel(
        [
            'build',
            '--nobuild',
            '--lockfile_mode=off',
            '//:all',
        ],
    )

    # Change registry -> update 'sss' module file (corrupt it)
    module_dir = self.main_registry.root.joinpath('modules', 'sss', '1.3')
    scratchFile(module_dir.joinpath('MODULE.bazel'), ['whatever!'])

    # Shutdown bazel to empty any cache of the deps tree
    self.RunBazel(['shutdown'])
    # Runing again will try to get 'sss' which should produce an error
    exit_code, _, stderr = self.RunBazel(
        [
            'build',
            '--nobuild',
            '--lockfile_mode=off',
            '//:all',
        ],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Error computing the main repository mapping: error parsing'
            ' MODULE.bazel file for sss@1.3'
        ),
        stderr,
    )

  def testChangeModuleInRegistryWithLockfile(self):
    # Add module 'sss' to the registry with dep on 'aaa'
    self.main_registry.createCcModule('sss', '1.3', {'aaa': '1.1'})
    # Create a project with deps on 'sss'
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "sss", version = "1.3")',
        ],
    )
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.RunBazel(
        [
            'build',
            '--nobuild',
            '//:all',
        ],
    )

    # Change registry -> update 'sss' module file (corrupt it)
    module_dir = self.main_registry.root.joinpath('modules', 'sss', '1.3')
    scratchFile(module_dir.joinpath('MODULE.bazel'), ['whatever!'])

    # Shutdown bazel to empty any cache of the deps tree
    self.RunBazel(['shutdown'])
    # Running with the lockfile, should not recognize the registry changes
    # hence find no errors
    self.RunBazel(['build', '--nobuild', '//:all'])

  def testChangeFlagWithLockfile(self):
    # Add module 'sss' to the registry with dep on 'aaa'
    self.main_registry.createCcModule('sss', '1.3', {'aaa': '1.1'})
    # Create a project with deps on 'sss'
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "sss", version = "1.3")',
        ],
    )
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.RunBazel(
        ['build', '--nobuild', '//:all'],
    )

    # Change registry -> update 'sss' module file (corrupt it)
    module_dir = self.main_registry.root.joinpath('modules', 'sss', '1.3')
    scratchFile(module_dir.joinpath('MODULE.bazel'), ['whatever!'])

    # Shutdown bazel to empty any cache of the deps tree
    self.RunBazel(['shutdown'])
    # Running with the lockfile, but adding a flag should cause resolution rerun
    exit_code, _, stderr = self.RunBazel(
        [
            'build',
            '--nobuild',
            '--check_direct_dependencies=error',
            '//:all',
        ],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        "ERROR: sss@1.3/MODULE.bazel:1:9: invalid character: '!'", stderr
    )

  def testLockfileErrorMode(self):
    self.ScratchFile('MODULE.bazel', [])
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.RunBazel(
        [
            'build',
            '--nobuild',
            '--check_direct_dependencies=warning',
            '//:all',
        ],
    )

    # Run with updated module and a different flag
    self.ScratchFile('MODULE.bazel', ['module(name="lala")'])
    exit_code, _, stderr = self.RunBazel(
        [
            'build',
            '--nobuild',
            '--check_direct_dependencies=error',
            '--lockfile_mode=error',
            '//:all',
        ],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Error computing the main repository mapping: Lock file is'
            ' no longer up-to-date because: the root MODULE.bazel has been'
            ' modified, the value of --check_direct_dependencies flag has'
            ' been modified'
        ),
        stderr,
    )

  def testLocalOverrideWithErrorMode(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'module(name="lala")',
            'bazel_dep(name="bar")',
            'local_path_override(module_name="bar",path="bar")',
        ],
    )
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.ScratchFile('bar/MODULE.bazel', ['module(name="bar")'])
    self.ScratchFile('bar/WORKSPACE', [])
    self.ScratchFile('bar/BUILD', ['filegroup(name = "hello from bar")'])
    self.RunBazel(
        [
            'build',
            '--nobuild',
            '//:all',
        ],
    )

    # Run with updated module and a different flag
    self.ScratchFile(
        'bar/MODULE.bazel',
        [
            'module(name="bar")',
            'bazel_dep(name="hmmm")',
        ],
    )
    exit_code, _, stderr = self.RunBazel(
        [
            'build',
            '--nobuild',
            '--lockfile_mode=error',
            '//:all',
        ],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Error computing the main repository mapping: Lock file is'
            ' no longer up-to-date because: The MODULE.bazel file has changed'
            ' for the overriden module: bar'
        ),
        stderr,
    )


if __name__ == '__main__':
  unittest.main()
