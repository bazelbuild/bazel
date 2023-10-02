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
from absl.testing import absltest
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
            'build --experimental_isolated_extension_usages',
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
            'ERROR: Error computing the main repository mapping: in module '
            'dependency chain <root> -> sss@1.3: error parsing MODULE.bazel '
            'file for sss@1.3'
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
            ' been modified. Please run'
            ' `bazel mod deps --lockfile_mode=update` to update your lockfile.'
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
            ' for the overriden module: bar. Please run'
            ' `bazel mod deps --lockfile_mode=update` to update your lockfile.'
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
            '_dep = tag_class(attrs={"name": attr.string(), "versions":',
            ' attr.string_list()})',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            '    tag_classes={"dep": _dep},',
            '    environ=["GREEN_TREES", "NOT_SET"],',
            '    os_dependent=True,',
            '    arch_dependent=True,',
            ')',
        ],
    )

    # Only set one env var, to make sure null variables don't crash
    _, _, stderr = self.RunBazel(
        ['build', '@hello//:all'], env_add={'GREEN_TREES': 'In the city'}
    )
    self.assertIn('Hello from the other side!', ''.join(stderr))
    self.assertIn('Name: bmbm , Versions: ["v1", "v2"]', ''.join(stderr))

    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(
        ['build', '@hello//:all'], env_add={'GREEN_TREES': 'In the city'}
    )
    self.assertNotIn('Hello from the other side!', ''.join(stderr))

  def testIsolatedAndNonIsolatedModuleExtensions(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            (
                'lockfile_ext_1 = use_extension("extension.bzl",'
                ' "lockfile_ext", isolate = True)'
            ),
            'use_repo(lockfile_ext_1, hello_1 = "hello")',
            'lockfile_ext_2 = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext_2, hello_2 = "hello")',
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
            '    print("Hello from the other side, %s!" % ctx.is_isolated)',
            '    repo_rule(name="hello")',
            '',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            ')',
        ],
    )

    _, _, stderr = self.RunBazel(['build', '@hello_1//:all', '@hello_2//:all'])
    self.assertIn('Hello from the other side, True!', ''.join(stderr))
    self.assertIn('Hello from the other side, False!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertIn(
          '//:extension.bzl%lockfile_ext%<root>~lockfile_ext_1',
          lockfile['moduleExtensions'],
      )
      self.assertIn(
          '//:extension.bzl%lockfile_ext', lockfile['moduleExtensions']
      )

    # Verify that the build succeeds using the lockfile.
    self.RunBazel(['shutdown'])
    self.RunBazel(['build', '@hello_1//:all', '@hello_2//:all'])

  def testUpdateIsolatedModuleExtension(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            (
                'lockfile_ext = use_extension("extension.bzl", "lockfile_ext",'
                ' isolate = True)'
            ),
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
            '    print("Hello from the other side, %s!" % ctx.is_isolated)',
            '    repo_rule(name="hello")',
            '',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            ')',
        ],
    )

    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side, True!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertIn(
          '//:extension.bzl%lockfile_ext%<root>~lockfile_ext',
          lockfile['moduleExtensions'],
      )
      self.assertNotIn(
          '//:extension.bzl%lockfile_ext', lockfile['moduleExtensions']
      )

    # Verify that the build succeeds using the lockfile.
    self.RunBazel(['shutdown'])
    self.RunBazel(['build', '@hello//:all'])

    # Update extension usage to be non-isolated.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext, "hello")',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side, False!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      self.assertNotIn(
          '//:extension.bzl%lockfile_ext%<root>~lockfile_ext',
          lockfile['moduleExtensions'],
      )
      self.assertIn(
          '//:extension.bzl%lockfile_ext', lockfile['moduleExtensions']
      )

    # Verify that the build succeeds using the lockfile.
    self.RunBazel(['shutdown'])
    self.RunBazel(['build', '@hello//:all'])

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
            'def _mod_ext_impl(ctx):',
            '    print("Hello from the other side!")',
            '    repo_rule(name= "hello")',
            'lockfile_ext = module_extension(implementation = _mod_ext_impl)',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other side!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      old_impl = lockfile['moduleExtensions']['//:extension.bzl%lockfile_ext'][
          'general'
      ]['bzlTransitiveDigest']

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
            'def _mod_ext_impl(ctx):',
            '    print("Hello from the other town!")',
            '    repo_rule(name= "hello")',
            'lockfile_ext = module_extension(implementation = _mod_ext_impl)',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    self.assertIn('Hello from the other town!', ''.join(stderr))
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      lockfile = json.loads(f.read().strip())
      new_impl = lockfile['moduleExtensions']['//:extension.bzl%lockfile_ext'][
          'general'
      ]['bzlTransitiveDigest']
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
            'def _mod_ext_impl(ctx):',
            '    print("Hello from the other side!")',
            '    repo_rule(name= "hello")',
            'lockfile_ext = module_extension(implementation = _mod_ext_impl)',
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
            'def _mod_ext_impl(ctx):',
            '    print("Hello from the other town!")',
            '    repo_rule(name= "hello")',
            'lockfile_ext = module_extension(implementation = _mod_ext_impl)',
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
            'of its transitive .bzl files has changed. Please run'
            ' `bazel mod deps --lockfile_mode=update` to update your lockfile.'
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

  def testNoAbsoluteRootModuleFilePath(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'ext = use_extension("extension.bzl", "ext")',
            'ext.dep(generate = True)',
            'use_repo(ext, ext_hello = "hello")',
            'other_ext = use_extension("extension.bzl", "other_ext")',
            'other_ext.dep(generate = False)',
            'use_repo(other_ext, other_ext_hello = "hello")',
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
            '    for mod in ctx.modules:',
            '        for dep in mod.tags.dep:',
            '            if dep.generate:',
            '                repo_rule(name="hello")',
            '',
            '_dep = tag_class(attrs={"generate": attr.bool()})',
            'ext = module_extension(',
            '    implementation=_module_ext_impl,',
            '    tag_classes={"dep": _dep},',
            ')',
            'other_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            '    tag_classes={"dep": _dep},',
            ')',
        ],
    )

    # Paths to module files in error message always use forward slashes as
    # separators, even on Windows.
    module_file_path = self.Path('MODULE.bazel').replace('\\', '/')

    self.RunBazel(['build', '--nobuild', '@ext_hello//:all'])
    with open(self.Path('MODULE.bazel.lock'), 'r') as f:
      self.assertNotIn(module_file_path, f.read())

    self.RunBazel(['shutdown'])
    exit_code, _, stderr = self.RunBazel(
        ['build', '--nobuild', '@other_ext_hello//:all'], allow_failure=True
    )
    self.AssertNotExitCode(exit_code, 0, stderr)
    self.assertIn(
        (
            'ERROR: module extension "other_ext" from "//:extension.bzl" does '
            'not generate repository "hello", yet it is imported as '
            '"other_ext_hello" in the usage at '
            + module_file_path
            + ':4:26'
        ),
        stderr,
    )

  def testModuleExtensionEnvVariable(self):
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
            'lockfile_ext = module_extension(',
            '    implementation = _module_ext_impl,',
            '    environ = ["SET_ME"]',
            ')',
        ],
    )
    _, _, stderr = self.RunBazel(
        ['build', '@hello//:all'], env_add={'SET_ME': 'High in sky'}
    )
    self.assertIn('Hello from the other side!', ''.join(stderr))
    # Run with same value, no evaluated
    _, _, stderr = self.RunBazel(
        ['build', '@hello//:all'], env_add={'SET_ME': 'High in sky'}
    )
    self.assertNotIn('Hello from the other side!', ''.join(stderr))
    # Run with different value, will be re-evaluated
    _, _, stderr = self.RunBazel(
        ['build', '@hello//:all'], env_add={'SET_ME': 'Down to earth'}
    )
    self.assertIn('Hello from the other side!', ''.join(stderr))

  def testChangeEnvVariableInErrorMode(self):
    # If environ is set up in module extension, it should be re-evaluated if its
    # value changed
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
            'lockfile_ext = module_extension(',
            '    implementation = _module_ext_impl,',
            '    environ = ["SET_ME"]',
            ')',
        ],
    )
    self.RunBazel(['build', '@hello//:all'], env_add={'SET_ME': 'High in sky'})
    exit_code, _, stderr = self.RunBazel(
        ['build', '--lockfile_mode=error', '@hello//:all'],
        env_add={'SET_ME': 'Down to earth'},
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Lock file is no longer up-to-date because: The environment'
            ' variables the extension'
            " 'ModuleExtensionId{bzlFileLabel=//:extension.bzl,"
            " extensionName=lockfile_ext, isolationKey=Optional.empty}' depends"
            ' on (or their values) have changed. Please run'
            ' `bazel mod deps --lockfile_mode=update` to update your lockfile.'
        ),
        stderr,
    )

  def testModuleExtensionWithFile(self):
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
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            '',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    print("I am running the extension")',
            '    print(ctx.read(Label("//:hello.txt")))',
            '    repo_rule(name="hello")',
            '',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl',
            ')',
        ],
    )

    self.ScratchFile('hello.txt', ['I will not stay the same.'])
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    stderr = ''.join(stderr)
    self.assertIn('I am running the extension', stderr)
    self.assertIn('I will not stay the same.', stderr)

    # Shutdown bazel to empty cache and run with no changes
    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    stderr = ''.join(stderr)
    self.assertNotIn('I am running the extension', stderr)
    self.assertNotIn('I will not stay the same.', stderr)

    # Update file and rerun
    self.ScratchFile('hello.txt', ['I have changed now!'])
    _, _, stderr = self.RunBazel(['build', '@hello//:all'])
    stderr = ''.join(stderr)
    self.assertIn('I am running the extension', stderr)
    self.assertIn('I have changed now!', stderr)

  def testOldVersion(self):
    self.ScratchFile('MODULE.bazel')
    self.ScratchFile('BUILD', ['filegroup(name = "hello")'])
    self.RunBazel(['build', '--nobuild', '//:all'])

    # Set version to old
    with open('MODULE.bazel.lock', 'r') as json_file:
      data = json.load(json_file)
      version = data['lockFileVersion']
    with open('MODULE.bazel.lock', 'w') as json_file:
      data['lockFileVersion'] = version - 1
      json.dump(data, json_file, indent=4)

    # Run in error mode
    exit_code, _, stderr = self.RunBazel(
        ['build', '--nobuild', '--lockfile_mode=error', '//:all'],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 48, stderr)
    self.assertIn(
        (
            'ERROR: Error computing the main repository mapping: Lock file is'
            ' no longer up-to-date because: the version of the lockfile is not'
            ' compatible with the current Bazel. Please run'
            ' `bazel mod deps --lockfile_mode=update` to update your lockfile.'
        ),
        stderr,
    )

    # Run again with update
    self.RunBazel(['build', '--nobuild', '//:all'])
    with open('MODULE.bazel.lock', 'r') as json_file:
      data = json.load(json_file)
    self.assertEqual(data['lockFileVersion'], version)

  def testExtensionEvaluationDoesNotRerunOnChangedImports(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext, "dep", "indirect_dep", "invalid_dep")',
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
            '    print("I am being evaluated")',
            '    repo_rule(name="dep")',
            '    repo_rule(name="missing_dep")',
            '    repo_rule(name="indirect_dep")',
            '    return ctx.extension_metadata(',
            '        root_module_direct_deps=["dep", "missing_dep"],',
            '        root_module_direct_dev_deps=[],',
            '    )',
            '',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl',
            ')',
        ],
    )

    # The build fails due to the "invalid_dep" import, which is not
    # generated by the extension.
    # Warnings should still be shown.
    _, _, stderr = self.RunBazel(['build', '@dep//:all'], allow_failure=True)
    stderr = '\n'.join(stderr)
    self.assertIn('I am being evaluated', stderr)
    self.assertIn(
        'Imported, but not created by the extension (will cause the build to'
        ' fail):\ninvalid_dep',
        stderr,
    )
    self.assertIn(
        'Not imported, but reported as direct dependencies by the extension'
        ' (may cause the build to fail):\nmissing_dep',
        stderr,
    )
    self.assertIn(
        'Imported, but reported as indirect dependencies by the'
        ' extension:\nindirect_dep',
        stderr,
    )
    self.assertIn(
        'ERROR: module extension "lockfile_ext" from "//:extension.bzl" does'
        ' not generate repository "invalid_dep"',
        stderr,
    )

    # Shut down bazel to empty cache and run with no changes to verify
    # that the warnings are still shown.
    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(['build', '@dep//:all'], allow_failure=True)
    stderr = '\n'.join(stderr)
    self.assertNotIn('I am being evaluated', stderr)
    self.assertIn(
        'Imported, but not created by the extension (will cause the build to'
        ' fail):\ninvalid_dep',
        stderr,
    )
    self.assertIn(
        'Not imported, but reported as direct dependencies by the extension'
        ' (may cause the build to fail):\nmissing_dep',
        stderr,
    )
    self.assertIn(
        'Imported, but reported as indirect dependencies by the'
        ' extension:\nindirect_dep',
        stderr,
    )
    self.assertIn(
        'ERROR: module extension "lockfile_ext" from "//:extension.bzl" does'
        ' not generate repository "invalid_dep"',
        stderr,
    )

    # Fix the imports, which should not trigger a rerun of the extension
    # even though imports and locations changed.
    self.ScratchFile(
        'MODULE.bazel',
        [
            '# This is a comment that changes the location of the usage below.',
            'lockfile_ext = use_extension("extension.bzl", "lockfile_ext")',
            'use_repo(lockfile_ext, "dep", "missing_dep")',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@dep//:all'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('I am being evaluated', stderr)
    self.assertNotIn(
        'Not imported, but reported as direct dependencies by the extension'
        ' (may cause the build to fail):\nmissing_dep',
        stderr,
    )
    self.assertNotIn(
        'Imported, but reported as indirect dependencies by the'
        ' extension:\nindirect_dep',
        stderr,
    )
    self.assertNotIn(
        'Imported, but reported as indirect dependencies by the'
        ' extension:\nindirect_dep',
        stderr,
    )
    self.assertNotIn(
        'ERROR: module extension "lockfile_ext" from "//:extension.bzl" does'
        ' not generate repository "invalid_dep"',
        stderr,
    )

  def testLockfileRecreatedAfterDeletion(self):
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
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            '',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    repo_rule(name="hello")',
            '',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl,',
            ')',
        ],
    )

    self.RunBazel(['build', '@hello//:all'])

    # Return the lockfile to the state it had before the
    # previous build: it didn't exist.
    with open('MODULE.bazel.lock', 'r') as lock_file:
      old_data = lock_file.read()
    os.remove('MODULE.bazel.lock')

    self.RunBazel(['build', '@hello//:all'])

    with open('MODULE.bazel.lock', 'r') as lock_file:
      new_data = lock_file.read()

    self.assertEqual(old_data, new_data)

  def testExtensionOsAndArch(self):
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
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    repo_rule(name="hello")',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl',
            ')',
        ],
    )

    # build to generate lockfile with this extension
    self.RunBazel(['build', '@hello//:all'])

    # validate an extension named 'general' is created
    general_key = 'general'
    ext_key = '//:extension.bzl%lockfile_ext'
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
      self.assertIn(ext_key, lockfile['moduleExtensions'])
      extension_map = lockfile['moduleExtensions'][ext_key]
      self.assertIn(general_key, extension_map)

    # replace general extension with one depend on os and arch
    win_key = 'os:WinWin,arch:arch32'
    with open('MODULE.bazel.lock', 'w') as json_file:
      extension_map[win_key] = extension_map.pop(general_key)
      json.dump(lockfile, json_file, indent=4)

    # update extension to depend on OS and arch.
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    repo_rule(name="hello")',
            'lockfile_ext = module_extension(',
            (
                'implementation=_module_ext_impl, os_dependent=True, '
                'arch_dependent=True'
            ),
            ')',
        ],
    )

    # build again to update the file
    self.RunBazel(['build', '@hello//:all'])
    # assert win_key still exists and another one was added
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
      extension_map = lockfile['moduleExtensions'][ext_key]
      self.assertIn(win_key, extension_map)
      self.assertEqual(len(extension_map), 2)
      added_key = ''
      for key in extension_map.keys():
        if key != win_key:
          added_key = key

    # update extension to only depend on os only
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "filegroup(name=\'lala\')")',
            'repo_rule = repository_rule(implementation=_repo_rule_impl)',
            '',
            'def _module_ext_impl(ctx):',
            '    repo_rule(name="hello")',
            'lockfile_ext = module_extension(',
            '    implementation=_module_ext_impl, os_dependent=True',
            ')',
        ],
    )

    # build again to update the file
    self.RunBazel(['build', '@hello//:all'])
    # assert both win_key and the added_key before are deleted,
    # and a new one without arch exists
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
      extension_map = lockfile['moduleExtensions'][ext_key]
      self.assertNotIn(win_key, extension_map)
      self.assertNotIn(added_key, extension_map)
      self.assertEqual(len(extension_map), 1)

  def testExtensionEvaluationOnlyRerunOnRelevantUsagesChanges(self):
    self.main_registry.createCcModule('aaa', '1.0')

    self.ScratchFile(
        'MODULE.bazel',
        [
            'ext_1 = use_extension("extension.bzl", "ext_1")',
            'ext_1.tag()',
            'use_repo(ext_1, ext_1_dep = "dep")',
            'ext_2 = use_extension("extension.bzl", "ext_2")',
            'ext_2.tag()',
            'use_repo(ext_2, ext_2_dep = "dep")',
            'ext_3 = use_extension("extension.bzl", "ext_3")',
            'use_repo(ext_3, ext_3_dep = "dep")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_rule_impl(ctx):',
            '    ctx.file("WORKSPACE")',
            '    ctx.file("BUILD", "exports_files([\\"data.txt\\"])")',
            '    ctx.file("data.txt", ctx.attr.value)',
            '    print(ctx.attr.value)',
            '',
            'repo_rule = repository_rule(',
            '    implementation=_repo_rule_impl,',
            '    attrs = {"value": attr.string()},)',
            '',
            'def _ext_1_impl(ctx):',
            '    print("Ext 1 is being evaluated")',
            '    num_tags = len([',
            '        tag for mod in ctx.modules for tag in mod.tags.tag',
            '    ])',
            '    repo_rule(name="dep", value="Ext 1 saw %s tags" % num_tags)',
            '',
            'ext_1 = module_extension(',
            '    implementation=_ext_1_impl,',
            '    tag_classes={"tag": tag_class()}',
            ')',
            '',
            'def _ext_2_impl(ctx):',
            '    print("Ext 2 is being evaluated")',
            '    num_tags = len([',
            '        tag for mod in ctx.modules for tag in mod.tags.tag',
            '    ])',
            '    repo_rule(name="dep", value="Ext 2 saw %s tags" % num_tags)',
            '',
            'ext_2 = module_extension(',
            '    implementation=_ext_2_impl,',
            '    tag_classes={"tag": tag_class()}',
            ')',
            '',
            'def _ext_3_impl(ctx):',
            '    print("Ext 3 is being evaluated")',
            '    num_tags = len([',
            '        tag for mod in ctx.modules for tag in mod.tags.tag',
            '    ])',
            '    repo_rule(name="dep", value="Ext 3 saw %s tags" % num_tags)',
            '',
            'ext_3 = module_extension(',
            '    implementation=_ext_3_impl,',
            '    tag_classes={"tag": tag_class()}',
            ')',
        ],
    )

    # Trigger evaluation of all extensions.
    _, _, stderr = self.RunBazel(
        ['build', '@ext_1_dep//:all', '@ext_2_dep//:all', '@ext_3_dep//:all']
    )
    stderr = '\n'.join(stderr)

    self.assertIn('Ext 1 is being evaluated', stderr)
    self.assertIn('Ext 1 saw 1 tags', stderr)
    self.assertIn('Ext 2 is being evaluated', stderr)
    self.assertIn('Ext 2 saw 1 tags', stderr)
    self.assertIn('Ext 3 is being evaluated', stderr)
    self.assertIn('Ext 3 saw 0 tags', stderr)
    ext_1_key = '//:extension.bzl%ext_1'
    ext_2_key = '//:extension.bzl%ext_2'
    ext_3_key = '//:extension.bzl%ext_3'
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
    self.assertIn(ext_1_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 1 saw 1 tags',
        lockfile['moduleExtensions'][ext_1_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )
    self.assertIn(ext_2_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 2 saw 1 tags',
        lockfile['moduleExtensions'][ext_2_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )
    self.assertIn(ext_3_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 3 saw 0 tags',
        lockfile['moduleExtensions'][ext_3_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )

    # Shut down bazel to empty the cache, modify the MODULE.bazel
    # file in a way that does not affect the resolution of ext_1,
    # but requires rerunning module resolution and removes ext_3, then
    # trigger module resolution without evaluating any of the extensions.
    self.RunBazel(['shutdown'])
    self.ScratchFile(
        'MODULE.bazel',
        [
            '# Added a dep to force rerunning module resolution.',
            'bazel_dep(name = "aaa", version = "1.0")',
            (
                '# The usage of ext_1 is unchanged except for locations and'
                ' imports.'
            ),
            'ext_1 = use_extension("extension.bzl", "ext_1")',
            'ext_1.tag()',
            'use_repo(ext_1, ext_1_dep_new_name = "dep")',
            '# The usage of ext_2 has a new tag.',
            'ext_2 = use_extension("extension.bzl", "ext_2")',
            'ext_2.tag()',
            'ext_2.tag()',
            'use_repo(ext_2, ext_2_dep = "dep")',
            '# The usage of ext_3 has been removed.',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '//:all'])
    stderr = '\n'.join(stderr)

    self.assertNotIn('Ext 1 is being evaluated', stderr)
    self.assertNotIn('Ext 2 is being evaluated', stderr)
    self.assertNotIn('Ext 3 is being evaluated', stderr)
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
    # The usages of ext_1 did not change.
    self.assertIn(ext_1_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 1 saw 1 tags',
        lockfile['moduleExtensions'][ext_1_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )
    # The usages of ext_2 changed, but the extension is not re-evaluated,
    # so its previous, now stale resolution result must have been removed.
    self.assertNotIn(ext_2_key, lockfile['moduleExtensions'])
    # The only usage of ext_3 was removed.
    self.assertNotIn(ext_3_key, lockfile['moduleExtensions'])

    # Trigger evaluation of all remaining extensions.
    _, _, stderr = self.RunBazel(
        ['build', '@ext_1_dep_new_name//:all', '@ext_2_dep//:all']
    )
    stderr = '\n'.join(stderr)

    self.assertNotIn('Ext 1 is being evaluated', stderr)
    self.assertIn('Ext 2 is being evaluated', stderr)
    self.assertIn('Ext 2 saw 2 tags', stderr)
    ext_1_key = '//:extension.bzl%ext_1'
    ext_2_key = '//:extension.bzl%ext_2'
    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
    self.assertIn(ext_1_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 1 saw 1 tags',
        lockfile['moduleExtensions'][ext_1_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )
    self.assertIn(ext_2_key, lockfile['moduleExtensions'])
    self.assertIn(
        'Ext 2 saw 2 tags',
        lockfile['moduleExtensions'][ext_2_key]['general'][
            'generatedRepoSpecs'
        ]['dep']['attributes']['value'],
    )
    self.assertNotIn(ext_3_key, lockfile['moduleExtensions'])

  def testLockfileWithNoUserSpecificPath(self):
    self.my_registry = BazelRegistry(os.path.join(self._test_cwd, 'registry'))
    patch_file = self.ScratchFile(
        'ss.patch',
        [
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
        ],
    )
    self.my_registry.createCcModule(
        'ss', '1.3-1', patches=[patch_file], patch_strip=1
    )

    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "ss", version = "1.3-1")',
        ],
    )
    self.ScratchFile('BUILD.bazel', ['filegroup(name = "lala")'])
    self.RunBazel(
        ['build', '--registry=file:///%workspace%/registry', '//:lala']
    )

    with open('MODULE.bazel.lock', 'r') as json_file:
      lockfile = json.load(json_file)
    remote_patches = lockfile['moduleDepGraph']['ss@1.3-1']['repoSpec'][
        'attributes'
    ]['remote_patches']
    for key in remote_patches.keys():
      self.assertIn('%workspace%', key)


if __name__ == '__main__':
  absltest.main()
