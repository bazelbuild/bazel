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

import json
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
    self.assertRegex(
        '\n'.join(stderr),
        "ERROR: .*/sss/1.3/MODULE.bazel:1:9: invalid character: '!'",
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
        ['build', '--nobuild', '--lockfile_mode=error', '//:all'],
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

  def testModuleExtension(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'lockfile_ext.dep(name = "bmbm", versions = ["v1", "v2"])',
            'use_repo(lockfile_ext, "hello")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            '',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    print("Hello from the other side!")',
            '    repo_rule(name="hello")',
            '    for mod in ctx.modules:',
            '        for dep in mod.tags.dep:',
            '            print("Name:", dep.name, ", Versions:", dep.versions)',
            '',
            (
                '_dep = tag_class(attrs={"name": attr.string(), "versions":'
                ' attr.string_list()})'
            ),
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            '    tag_classes={"dep": _dep},',
            ')',
        ],
    )

    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side!', ''.join(stderr))
    self.assertIn('Name: bmbm , Versions: ["v1", "v2"]', ''.join(stderr))

    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertNotIn('Hello from the other side!', ''.join(stderr))

  def testModuleExtensionsInDifferentBuilds(self):
    # Test that the module extension stays in the lockfile (as long as it's
    # used in the module) even if it is not in the current build
    self.ScratchFile(
        'MODULE.bazel',
        [
            'extA = use_extension("extension.bzl", "extA")',
            'extB = use_extension("extension.bzl", "extB")',
            'use_repo(extA, "hello_ext_A")',
            'use_repo(extB, "hello_ext_B")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            'def _ext_a_impl(ctx):',
            '    repo_rule(name="hello_ext_A")',
            'def _ext_b_impl(ctx):',
            '    repo_rule(name="hello_ext_B")',
            'extA = module_extension(implementation=_ext_a_impl)',
            'extB = module_extension(implementation=_ext_b_impl)',
        ],
    )

    self.RunBazel(['build', '@hello_ext_A//:all'])
    self.RunBazel(['build', '@hello_ext_B//:all'])

    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertGreater(len(lockfile['moduleDepGraph']), 0)
      self.assertEqual(
          list(lockfile['moduleExtensions'].keys()),
          ['//:extension.bzl%extA', '//:extension.bzl%extB'],
      )

  def testUpdateModuleExtension(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext, "hello")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\\"lala\\")")',
            'repo_rule = repository_rule(implementation = _repo_rule_impl)',
            'def _module_ext_impl(ctx):',
            '    print("Hello from the other side!")',
            '    repo_rule(name= "hello")',
            (
                'lockfile_ext = module_extension(implementation ='
                ' _module_ext_impl)'
            ),
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      old_impl = lockfile['moduleExtensions']['//:extension.bzl%lockfile_ext'][
          'bzlTransitiveDigest'
      ]

    # Run again to make sure the resolution value is cached. So even if module
    # resolution doesn't rerun (its event is null), the lockfile is still
    # updated with the newest extension eval results
    self.RunBazel(['build', '@hello//:all'])

    # Update extension. Make sure that it is executed and updated in the
    # lockfile without errors (since it's already in the lockfile)
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\\"lala\\")")',
            'repo_rule = repository_rule(implementation = _repo_rule_impl)',
            'def _module_ext_impl(ctx):',
            '    print("Hello from the other town!")',
            '    repo_rule(name= "hello")',
            (
                'lockfile_ext = module_extension(implementation ='
                ' _module_ext_impl)'
            ),
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other town!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      new_impl = lockfile['moduleExtensions']['//:extension.bzl%lockfile_ext'][
          'bzlTransitiveDigest'
      ]
      self.assertNotEqual(new_impl, old_impl)

  def testUpdateModuleExtensionErrorMode(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext, "hello")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\\"lala\\")")',
            'repo_rule = repository_rule(implementation = _repo_rule_impl)',
            'def _module_ext_impl(ctx):',
            '    print("Hello from the other side!")',
            '    repo_rule(name= "hello")',
            (
                'lockfile_ext = module_extension(implementation ='
                ' _module_ext_impl)'
            ),
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side!', ''.join(stderr))
    self.RunBazel(['shutdown'])

    # Update extension.
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\\"lalo\\")")',
            'repo_rule = repository_rule(implementation = _repo_rule_impl)',
            'def _module_ext_impl(ctx):',
            '    print("Hello from the other town!")',
            '    repo_rule(name= "hello")',
            (
                'lockfile_ext = module_extension(implementation ='
                ' _module_ext_impl)'
            ),
        ],
    )

    exit_code, _, stderr = self.RunBazel(
        ['build', '--nobuild', '--lockfile_mode=error', '@hello//:all'],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Lock file is no longer up-to-date because: The '
            'implementation of the extension '
            "'ModuleExtensionId{bzlFileLabel=//:extension.bzl, "
            "extensionName=lockfile_ext, isolationKey=Optional.empty}' or one "
            'of its transitive .bzl files has changed'
        ),
        stderr,
    )

  def testRemoveModuleExtensionsNotUsed(self):
    # Test that the module extension is removed from the lockfile if it is not
    # used anymore
    self.ScratchFile(
        'MODULE.bazel',
        [
            'ext = use_extension("extension.bzl", "ext")',
            'use_repo(ext, "hello")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            'def _ext_impl(ctx):',
            '    repo_rule(name="hello")',
            'ext = module_extension(implementation=_ext_impl)',
        ],
    )

    self.RunBazel(['build', '@hello//:all'])
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertEqual(
          list(lockfile['moduleExtensions'].keys()), ['//:extension.bzl%ext']
      )

    self.ScratchFile('MODULE.bazel', [])
    self.RunBazel(['build', '//:all'])
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertEqual(len(lockfile['moduleExtensions']), 0)


if __name__ == '__main__':
  unittest.main()
