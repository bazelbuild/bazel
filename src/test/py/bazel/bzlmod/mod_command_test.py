# pylint: disable=g-backslash-continuation
# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Tests the mod command."""

import os
import tempfile
import unittest

from src.test.py.bazel import test_base
from src.test.py.bazel.bzlmod.test_utils import BazelRegistry
from src.test.py.bazel.bzlmod.test_utils import scratchFile


class ModCommandTest(test_base.TestBase):
  """Test class for the mod command."""

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.registries_work_dir = tempfile.mkdtemp(dir=self._test_cwd)
    self.main_registry = BazelRegistry(
        os.path.join(self.registries_work_dir, 'main')
    )
    self.main_registry.setModuleBasePath('projects')
    self.projects_dir = self.main_registry.projects
    self.maxDiff = None  # there are some long diffs in this test

    self.ScratchFile(
        '.bazelrc',
        [
            # In ipv6 only network, this has to be enabled.
            # 'startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true',
            'mod --enable_bzlmod',
            'mod --registry=' + self.main_registry.getURL(),
            # We need to have BCR here to make sure built-in modules like
            # bazel_tools can work.
            'mod --registry=https://bcr.bazel.build',
            # Disable yanked version check so we are not affected BCR changes.
            'mod --allow_yanked_versions=all',
            # Make sure Bazel CI tests pass in all environments
            'mod --charset=ascii',
        ],
    )
    self.ScratchFile('WORKSPACE')
    # The existence of WORKSPACE.bzlmod prevents WORKSPACE prefixes or suffixes
    # from being used; this allows us to test built-in modules actually work
    self.ScratchFile('WORKSPACE.bzlmod')

    self.ScratchFile(
        'MODULE.bazel',
        [
            'module(name = "my_project", version = "1.0")',
            '',
            'bazel_dep(name = "foo", version = "1.0", repo_name = "foo1")',
            'bazel_dep(name = "foo", version = "2.0", repo_name = "foo2")',
            'bazel_dep(name = "ext", version = "1.0")',
            'bazel_dep(name = "ext2", version = "1.0")',
            'multiple_version_override(',
            '  module_name= "foo",',
            '  versions = ["1.0", "2.0"],',
            ')',
            'ext = use_extension("@ext//:ext.bzl", "ext")',
            'use_repo(ext, myrepo="repo1")',
            'ext2 = use_extension("@ext2//:ext.bzl", "ext")',
            'ext2.dep(name="repo1")',
            'use_repo(ext2, myrepo2="repo1")',
        ],
    )
    self.main_registry.createCcModule(
        'foo',
        '1.0',
        {'bar': '1.0', 'ext': '1.0'},
        {'bar': 'bar_from_foo1'},
        extra_module_file_contents=[
            'my_ext = use_extension("@ext//:ext.bzl", "ext")',
            'my_ext.dep(name="repo1")',
            'my_ext.dep(name="repo2")',
            'my_ext.dep(name="repo5")',
            'use_repo(my_ext, my_repo1="repo1")',
        ],
    )
    self.main_registry.createCcModule(
        'foo',
        '2.0',
        {'bar': '2.0', 'ext': '1.0'},
        {'bar': 'bar_from_foo2', 'ext': 'ext_mod'},
        extra_module_file_contents=[
            'my_ext = use_extension("@ext_mod//:ext.bzl", "ext")',
            'my_ext.dep(name="repo4")',
            'use_repo(my_ext, my_repo3="repo3", my_repo4="repo4")',
        ],
    )
    self.main_registry.createCcModule('bar', '1.0', {'ext': '1.0'})
    self.main_registry.createCcModule(
        'bar',
        '2.0',
        {'ext': '1.0', 'ext2': '1.0'},
        extra_module_file_contents=[
            'my_ext = use_extension("@ext//:ext.bzl", "ext")',
            'my_ext.dep(name="repo3")',
            'use_repo(my_ext, my_repo3="repo3")',
            'my_ext2 = use_extension("@ext2//:ext.bzl", "ext")',
            'my_ext2.dep(name="repo3")',
            'use_repo(my_ext2, my_repo2="repo3")',
        ],
    )

    ext_src = [
        'def _data_repo_impl(ctx): ctx.file("BUILD")',
        'data_repo = repository_rule(_data_repo_impl,',
        '  attrs={"data":attr.string()},',
        ')',
        'def _ext_impl(ctx):',
        '  deps = {dep.name: 1 for mod in ctx.modules for dep in mod.tags.dep}',
        '  for dep in deps:',
        '    data_repo(name=dep, data="requested repo")',
        'ext=module_extension(_ext_impl,',
        '  tag_classes={"dep":tag_class(attrs={"name":attr.string()})},',
        ')',
    ]

    self.main_registry.createLocalPathModule('ext', '1.0', 'ext')
    self.projects_dir.joinpath('ext').mkdir(exist_ok=True)
    scratchFile(self.projects_dir.joinpath('ext', 'WORKSPACE'))
    scratchFile(self.projects_dir.joinpath('ext', 'BUILD'))
    scratchFile(self.projects_dir.joinpath('ext', 'ext.bzl'), ext_src)
    self.main_registry.createLocalPathModule('ext2', '1.0', 'ext2')
    self.projects_dir.joinpath('ext2').mkdir(exist_ok=True)
    scratchFile(self.projects_dir.joinpath('ext2', 'WORKSPACE'))
    scratchFile(self.projects_dir.joinpath('ext2', 'BUILD'))
    scratchFile(self.projects_dir.joinpath('ext2', 'ext.bzl'), ext_src)

  def testGraph(self):
    _, stdout, _ = self.RunBazel(['mod', 'graph'], rstrip=True)
    self.assertListEqual(
        stdout,
        [
            '<root> (my_project@1.0)',
            '|___ext@1.0',
            '|___ext2@1.0',
            '|___foo@1.0',
            '|   |___ext@1.0 (*)',
            '|   |___bar@2.0',
            '|       |___ext@1.0 (*)',
            '|       |___ext2@1.0 (*)',
            '|___foo@2.0',
            '    |___bar@2.0 (*)',
            '    |___ext@1.0 (*)',
            '',
        ],
        'wrong output in graph query',
    )

  def testGraphWithExtensions(self):
    _, stdout, _ = self.RunBazel(
        ['mod', 'graph', '--extension_info=all'], rstrip=True
    )
    self.assertListEqual(
        stdout,
        [
            '<root> (my_project@1.0)',
            '|___$@@ext2~1.0//:ext.bzl%ext',
            '|   |___repo1',
            '|___$@@ext~1.0//:ext.bzl%ext',
            '|   |___repo1',
            '|   |...repo2',
            '|   |...repo5',
            '|___ext@1.0',
            '|___ext2@1.0',
            '|___foo@1.0',
            '|   |___$@@ext~1.0//:ext.bzl%ext ...',
            '|   |   |___repo1',
            '|   |___ext@1.0 (*)',
            '|   |___bar@2.0',
            '|       |___$@@ext2~1.0//:ext.bzl%ext ...',
            '|       |   |___repo3',
            '|       |___$@@ext~1.0//:ext.bzl%ext ...',
            '|       |   |___repo3',
            '|       |___ext@1.0 (*)',
            '|       |___ext2@1.0 (*)',
            '|___foo@2.0',
            '    |___$@@ext~1.0//:ext.bzl%ext ...',
            '    |   |___repo3',
            '    |   |___repo4',
            '    |___bar@2.0 (*)',
            '    |___ext@1.0 (*)',
            '',
        ],
        'wrong output in graph with extensions query',
    )

  def testGraphWithExtensionFilter(self):
    _, stdout, _ = self.RunBazel(
        [
            'mod',
            'graph',
            '--extension_info=repos',
            '--extension_filter=@ext//:ext.bzl%ext',
        ],
        rstrip=True,
    )
    self.assertListEqual(
        stdout,
        [
            '<root> (my_project@1.0)',
            '|___$@@ext~1.0//:ext.bzl%ext',
            '|   |___repo1',
            '|___foo@1.0 #',
            '|   |___$@@ext~1.0//:ext.bzl%ext',
            '|   |   |___repo1',
            '|   |___bar@2.0 #',
            '|       |___$@@ext~1.0//:ext.bzl%ext',
            '|           |___repo3',
            '|___foo@2.0 #',
            '    |___$@@ext~1.0//:ext.bzl%ext',
            '    |   |___repo3',
            '    |   |___repo4',
            '    |___bar@2.0 (*)',
            '',
        ],
        'wrong output in graph query with extension filter specified',
    )

  def testShowExtensionAllUsages(self):
    _, stdout, _ = self.RunBazel(
        ['mod', 'show_extension', '@ext//:ext.bzl%ext'], rstrip=True
    )
    self.assertRegex(
        stdout.pop(9), r'^## Usage in <root> from .*MODULE\.bazel:11$'
    )
    self.assertRegex(
        stdout.pop(14), r'^## Usage in foo@1.0 from .*MODULE\.bazel:8$'
    )
    self.assertRegex(
        stdout.pop(22), r'^## Usage in foo@2.0 from .*MODULE\.bazel:8$'
    )
    self.assertRegex(
        stdout.pop(29), r'^## Usage in bar@2.0 from .*MODULE\.bazel:8$'
    )
    self.assertListEqual(
        stdout,
        [
            '## @@ext~1.0//:ext.bzl%ext:',
            '',
            'Fetched repositories:',
            '  - repo1 (imported by <root>, foo@1.0)',
            '  - repo3 (imported by bar@2.0, foo@2.0)',
            '  - repo4 (imported by foo@2.0)',
            '  - repo2',
            '  - repo5',
            '',
            # pop(9)
            'use_repo(',
            '  ext,',
            '  myrepo="repo1",',
            ')',
            '',
            # pop(14)
            'ext.dep(name="repo1")',
            'ext.dep(name="repo2")',
            'ext.dep(name="repo5")',
            'use_repo(',
            '  ext,',
            '  my_repo1="repo1",',
            ')',
            '',
            # pop(22)
            'ext.dep(name="repo4")',
            'use_repo(',
            '  ext,',
            '  my_repo3="repo3",',
            '  my_repo4="repo4",',
            ')',
            '',
            # pop(29)
            'ext.dep(name="repo3")',
            'use_repo(',
            '  ext,',
            '  my_repo3="repo3",',
            ')',
            '',
        ],
        'wrong output in show_extension query with all usages',
    )

  def testShowExtensionSomeExtensionsSomeUsages(self):
    _, stdout, _ = self.RunBazel(
        [
            'mod',
            'show_extension',
            '@ext//:ext.bzl%ext',
            '@ext2//:ext.bzl%ext',
            '--extension_usages=@foo2,bar@2.0',
        ],
        rstrip=True,
    )
    self.assertRegex(
        stdout.pop(6), r'^## Usage in bar@2.0 from .*MODULE\.bazel:11$'
    )
    self.assertRegex(
        stdout.pop(21), r'^## Usage in foo@2.0 from .*MODULE\.bazel:8$'
    )
    self.assertRegex(
        stdout.pop(28), r'^## Usage in bar@2.0 from .*MODULE\.bazel:8$'
    )
    self.assertListEqual(
        stdout,
        [
            '## @@ext2~1.0//:ext.bzl%ext:',
            '',
            'Fetched repositories:',
            '  - repo1 (imported by <root>)',
            '  - repo3 (imported by bar@2.0)',
            '',
            # pop(6)
            'ext.dep(name="repo3")',
            'use_repo(',
            '  ext,',
            '  my_repo2="repo3",',
            ')',
            '',
            '## @@ext~1.0//:ext.bzl%ext:',
            '',
            'Fetched repositories:',
            '  - repo1 (imported by <root>, foo@1.0)',
            '  - repo3 (imported by bar@2.0, foo@2.0)',
            '  - repo4 (imported by foo@2.0)',
            '  - repo2',
            '  - repo5',
            '',
            # pop(21)
            'ext.dep(name="repo4")',
            'use_repo(',
            '  ext,',
            '  my_repo3="repo3",',
            '  my_repo4="repo4",',
            ')',
            '',
            # pop(28)
            'ext.dep(name="repo3")',
            'use_repo(',
            '  ext,',
            '  my_repo3="repo3",',
            ')',
            '',
        ],
        'Wrong output in the show with some extensions and some usages query.',
    )

  def testShowModuleAndExtensionReposFromBaseModule(self):
    _, stdout, _ = self.RunBazel(
        [
            'mod',
            'show_repo',
            '--base_module=foo@2.0',
            '@bar_from_foo2',
            'ext@1.0',
            '@my_repo3',
            '@my_repo4',
            'bar',
        ],
        rstrip=True,
    )
    self.assertRegex(stdout.pop(4), r'^  urls = \[".*"\],$')
    self.assertRegex(stdout.pop(4), r'^  integrity = ".*",$')
    stdout.pop(11)
    self.assertRegex(stdout.pop(16), r'^  path = ".*",$')
    stdout.pop(29)
    stdout.pop(39)
    self.assertRegex(stdout.pop(44), r'^  urls = \[".*"\],$')
    self.assertRegex(stdout.pop(44), r'^  integrity = ".*",$')
    stdout.pop(51)
    self.assertListEqual(
        stdout,
        [
            '## @bar_from_foo2:',
            '# <builtin>',
            'http_archive(',
            '  name = "bar~2.0",',
            # pop(4) -- urls=[...]
            # pop(4) -- integrity=...
            '  strip_prefix = "",',
            '  remote_patches = {},',
            '  remote_patch_strip = 0,',
            ')',
            '# Rule bar~2.0 instantiated at (most recent call last):',
            '#   <builtin> in <toplevel>',
            '# Rule http_archive defined at (most recent call last):',
            # pop(11)
            '',
            '## ext@1.0:',
            '# <builtin>',
            'local_repository(',
            '  name = "ext~1.0",',
            # pop(16) -- path=...
            ')',
            '# Rule ext~1.0 instantiated at (most recent call last):',
            '#   <builtin> in <toplevel>',
            '',
            '## @my_repo3:',
            '# <builtin>',
            'data_repo(',
            '  name = "ext~1.0~ext~repo3",',
            '  data = "requested repo",',
            ')',
            '# Rule ext~1.0~ext~repo3 instantiated at (most recent call last):',
            '#   <builtin> in <toplevel>',
            '# Rule data_repo defined at (most recent call last):',
            # pop(29)
            '',
            '## @my_repo4:',
            '# <builtin>',
            'data_repo(',
            '  name = "ext~1.0~ext~repo4",',
            '  data = "requested repo",',
            ')',
            '# Rule ext~1.0~ext~repo4 instantiated at (most recent call last):',
            '#   <builtin> in <toplevel>',
            '# Rule data_repo defined at (most recent call last):',
            # pop(39)
            '',
            '## bar@2.0:',
            '# <builtin>',
            'http_archive(',
            '  name = "bar~2.0",',
            # pop(44) -- urls=[...]
            # pop(44) -- integrity=...
            '  strip_prefix = "",',
            '  remote_patches = {},',
            '  remote_patch_strip = 0,',
            ')',
            '# Rule bar~2.0 instantiated at (most recent call last):',
            '#   <builtin> in <toplevel>',
            '# Rule http_archive defined at (most recent call last):',
            # pop(51)
            '',
        ],
        'wrong output in the show query for module and extension-generated'
        ' repos',
    )

  def testShowRepoThrowsUnusedModule(self):
    _, _, stderr = self.RunBazel(
        ['mod', 'show_repo', 'bar@1.0', '--base_module=@foo2'],
        allow_failure=True,
        rstrip=True,
    )
    self.assertIn(
        'ERROR: In repo argument bar@1.0: Module version bar@1.0 does not'
        ' exist, available versions: [bar@2.0]. (Note that unused modules'
        " cannot be used here). Type 'bazel help mod' for syntax and help.",
        stderr,
    )


if __name__ == '__main__':
  unittest.main()
