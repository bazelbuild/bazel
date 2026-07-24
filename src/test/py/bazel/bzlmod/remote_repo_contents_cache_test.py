# pylint: disable=g-backslash-continuation
# Copyright 2025 The Bazel Authors. All rights reserved.
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
# pylint: disable=g-bad-todo

import json
import os
import re
import tempfile
from absl.testing import absltest
from src.test.py.bazel import test_base

# Whether repos containing symlinks that point out of the repo can be added to
# the remote repo contents cache. If False, such repos are refetched instead of
# being restored from the cache.
# TODO(#30160): Flip to True once the overlay file system supports resolving
# symlinks across its backing file systems.
CROSS_REPO_SYMLINKS_CACHEABLE = False


class RemoteRepoContentsCacheTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self._worker_port = self.StartRemoteWorker()
    self.ScratchFile(
        '.bazelrc',
        [
            'startup --experimental_remote_repo_contents_cache',
            # Only use the remote repo contents cache.
            'common --repo_contents_cache=',
            'common --remote_cache=grpc://localhost:' + str(self._worker_port),
            'common --auth_enabled=false',
            'common --remote_timeout=3600s',
            'common --verbose_failures',
        ],
    )

  def tearDown(self):
    test_base.TestBase.tearDown(self)
    self.StopRemoteWorker()

  def RepoDir(self, repo_name, cwd=None):
    _, stdout, _ = self.RunBazel(['info', 'output_base'], cwd=cwd)
    self.assertLen(stdout, 1)
    output_base = stdout[0].strip()

    _, stdout, _ = self.RunBazel(['mod', 'dump_repo_mapping', ''], cwd=cwd)
    self.assertLen(stdout, 1)
    mapping = json.loads(stdout[0])
    canonical_repo_name = mapping[repo_name]

    return output_base + '/external/' + canonical_repo_name

  def testCachedAfterCleanExpunge(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging, without using repo contents cache: not cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel([
        '--noexperimental_remote_repo_contents_cache',
        'build',
        '@my_repo//:haha',
    ])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testLocalRepoContentsCacheInteraction(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    repo_contents_cache = tempfile.mkdtemp(dir=os.environ['TEST_TMPDIR'])
    _, _, stderr = self.RunBazel([
        'build',
        '@my_repo//:haha',
        '--repo_contents_cache=' + repo_contents_cache,
    ])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging: cached, hits the local repo contents cache
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel([
        'build',
        '@my_repo//:haha',
        '--repo_contents_cache=' + repo_contents_cache,
    ])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After cleaning out local repo contents cache: cached, hits the remote
    # cache
    self.RunBazel(['clean', '--expunge'])
    # Deleting the cache fails on Windows, so we just use a different directory.
    _, _, stderr = self.RunBazel([
        'build',
        '@my_repo//:haha',
        '--repo_contents_cache=' + repo_contents_cache + '2',
    ])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging, without using any repo contents cache: not cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel([
        '--noexperimental_remote_repo_contents_cache',
        'build',
        '@my_repo//:haha',
    ])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testNotCachedWhenPredeclaredInputsChange(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo", data = 1)',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl, attrs={"data":attr.int()})',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change predeclared inputs: not cached
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo", data = 2)',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change back to previous predeclared inputs: cached (even after expunging)
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo", data = 1)',
        ],
    )
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testNotCachedWhenRecordedInputsChange_dynamicDep(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  rctx.watch(Label("@//:data.txt"))',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    self.ScratchFile('data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change recorded inputs: not cached
    self.ScratchFile('data.txt', ['two'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change back to previous recorded inputs: cached (even after expunging)
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testNotCachedWhenRecordedInputsChange_staticDep(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  rctx.os.environ.get("LOLOL")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl, environ = ["LOLOL"])',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'lol'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change recorded inputs: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'kek'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Change back to previous recorded inputs: cached (even after expunging)
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'lol'}
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testRecordedInputs_differentValues(self):
    platform_file = self.ScratchFile('platform.txt')

    self.ScratchFile(
        'MODULE.bazel',
        [
            (
                'platform_dependent_repo ='
                ' use_repo_rule("//:platform_dependent_repo.bzl",'
                ' "platform_dependent_repo")'
            ),
            'platform_dependent_repo(name = "platform_dependent_repo")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile(
        'BUILD.bazel',
        [
            'genrule(',
            '  name = "show_platform",',
            '  outs = ["platform.txt"],',
            '  cmd = "cat $(location @my_repo//:data.txt) > $@",',
            '  srcs = ["@my_repo//:data.txt"],',
            ')',
        ],
    )
    self.ScratchFile(
        'platform_dependent_repo.bzl',
        [
            'def _platform_dependent_repo_impl(rctx):',
            '  rctx.file("BUILD")',
            '  print("DETERMINING PLATFORM")',
            '  platform = rctx.read(rctx.path("%s"))'
            % platform_file.replace('\\', '\\\\'),
            '  rctx.file("data.txt", platform)',
            (
                'platform_dependent_repo ='
                ' repository_rule(_platform_dependent_repo_impl)'
            ),
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'data.txt\'])")',
            (
                '  platform ='
                ' rctx.read(Label("@platform_dependent_repo//:data.txt"))'
            ),
            '  print("JUST FETCHED ON " + platform)',
            '  rctx.file("data.txt", platform)',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch on Linux: not cached
    self.ScratchFile('platform.txt', ['Linux'])
    _, _, stderr = self.RunBazel(['build', '//:show_platform'])
    self.assertIn('DETERMINING PLATFORM', '\n'.join(stderr))
    self.assertIn('JUST FETCHED ON Linux', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/platform.txt')) as f:
      self.assertEqual(f.read().strip(), 'Linux')

    # First fetch on macOS: not cached
    self.ScratchFile('platform.txt', ['macOS'])
    _, _, stderr = self.RunBazel(['build', '//:show_platform'])
    self.assertIn('DETERMINING PLATFORM', '\n'.join(stderr))
    self.assertIn('JUST FETCHED ON macOS', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/platform.txt')) as f:
      self.assertEqual(f.read().strip(), 'macOS')

    # Second fetch on Linux: cached
    self.ScratchFile('platform.txt', ['Linux'])
    _, _, stderr = self.RunBazel(['build', '//:show_platform'])
    self.assertIn('DETERMINING PLATFORM', '\n'.join(stderr))
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/platform.txt')) as f:
      self.assertEqual(f.read().strip(), 'Linux')

    # Second fetch on macOS: cached
    self.ScratchFile('platform.txt', ['macOS'])
    _, _, stderr = self.RunBazel(['build', '//:show_platform'])
    self.assertIn('DETERMINING PLATFORM', '\n'.join(stderr))
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/platform.txt')) as f:
      self.assertEqual(f.read().strip(), 'macOS')

  def testRecordedInputs_differentInputs(self):
    platform_file = self.ScratchFile('platform.txt')

    self.ScratchFile(
        'MODULE.bazel',
        [
            (
                'platform_dependent_binary ='
                ' use_repo_rule("//:platform_dependent_binary.bzl",'
                ' "platform_dependent_binary")'
            ),
            'platform_dependent_binary(name = "platform_dependent_binary")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile(
        'BUILD.bazel',
        [
            'genrule(',
            '  name = "show_data",',
            '  outs = ["data.txt"],',
            '  cmd = "cat $(location @my_repo//:data.txt) > $@",',
            '  srcs = ["@my_repo//:data.txt"],',
            ')',
        ],
    )
    self.ScratchFile(
        'platform_dependent_binary.bzl',
        [
            'def _platform_dependent_binary_impl(rctx):',
            '  rctx.file("BUILD")',
            '  platform = rctx.read(rctx.path("%s")).strip()'
            % platform_file.replace('\\', '\\\\'),
            '  print("DETERMINED PLATFORM (%s)" % platform)',
            '  if platform == "Windows":',
            '    rctx.file("binary.exe", "PE")',
            '  else:',
            '    rctx.file("binary.sh", "ELF")',
            (
                'platform_dependent_binary ='
                ' repository_rule(_platform_dependent_binary_impl)'
            ),
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'data.txt\'])")',
            # Simulate a uname -s by reading from a file instead of using
            # rctx.execute (more complex to mock) or rctx.os.name (which may be
            # tracked as an input in the future, making this test vacuous).
            '  platform = rctx.read(rctx.path("%s"), watch = "no").strip()'
            % platform_file.replace('\\', '\\\\'),
            '  ext = ".exe" if platform == "Windows" else ".sh"',
            # Simulate rctx.execute with a watched binary.
            (
                '  out = rctx.read(Label("@platform_dependent_binary//:binary"'
                ' + ext))'
            ),
            '  rctx.file("data.txt", out)',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch on Linux: not cached
    self.ScratchFile('platform.txt', ['Linux'])
    _, _, stderr = self.RunBazel(['build', '//:show_data'])
    self.assertIn('DETERMINED PLATFORM (Linux)', '\n'.join(stderr))
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/data.txt')) as f:
      self.assertEqual(f.read().strip(), 'ELF')

    # First fetch on Windows: not cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('platform.txt', ['Windows'])
    _, _, stderr = self.RunBazel(['build', '//:show_data'])
    self.assertIn('DETERMINED PLATFORM (Windows)', '\n'.join(stderr))
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/data.txt')) as f:
      self.assertEqual(f.read().strip(), 'PE')

    # Second fetch on Linux: cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('platform.txt', ['Linux'])
    _, _, stderr = self.RunBazel(['build', '//:show_data'])
    self.assertIn('DETERMINED PLATFORM (Linux)', '\n'.join(stderr))
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/data.txt')) as f:
      self.assertEqual(f.read().strip(), 'ELF')

    # Second fetch on Windows: cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('platform.txt', ['Windows'])
    _, _, stderr = self.RunBazel(['build', '//:show_data'])
    self.assertIn('DETERMINED PLATFORM (Windows)', '\n'.join(stderr))
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    with open(self.Path('bazel-bin/data.txt')) as f:
      self.assertEqual(f.read().strip(), 'PE')

  def testNoThrashingBetweenWorkspaces(self):
    module_bazel_lines = [
        'repo = use_repo_rule("//:repo.bzl", "repo")',
        'repo(name = "my_repo")',
    ]
    repo_bzl_lines = [
        'def _repo_impl(rctx):',
        '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
        '  rctx.os.environ.get("LOLOL")',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl, environ = ["LOLOL"])',
    ]
    # Set up two workspaces with exactly the same repo definition (same
    # predeclared inputs)
    dir_a = self.ScratchDir('a')
    dir_b = self.ScratchDir('b')
    self.ScratchFile('a/MODULE.bazel', module_bazel_lines)
    self.ScratchFile('b/MODULE.bazel', module_bazel_lines)
    self.ScratchFile('a/BUILD.bazel')
    self.ScratchFile('b/BUILD.bazel')
    self.ScratchFile('a/repo.bzl', repo_bzl_lines)
    self.ScratchFile('b/repo.bzl', repo_bzl_lines)
    self.CopyFile(self.Path('.bazelrc'), 'a/.bazelrc')
    self.CopyFile(self.Path('.bazelrc'), 'b/.bazelrc')

    repo_dir_a = self.RepoDir('my_repo', cwd=dir_a)
    repo_dir_b = self.RepoDir('my_repo', cwd=dir_b)

    # First fetch in A: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], cwd=dir_a, env_add={'LOLOL': 'lol'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir_a, 'BUILD')))

    # Fetch in B (with same env variable): cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], cwd=dir_b, env_add={'LOLOL': 'lol'}
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir_b, 'BUILD')))

    # Change env variable: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], cwd=dir_b, env_add={'LOLOL': 'rofl'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir_b, 'BUILD')))

    # Building A again even after expunging: cached
    self.RunBazel(['clean', '--expunge'], cwd=dir_a)
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], cwd=dir_a, env_add={'LOLOL': 'rofl'}
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir_a, 'BUILD')))

  def testAccessFromOtherRepo_read(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", build_file = "@my_repo//:BUILD")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            # Verify that directories are materialized correctly.
            '  rctx.file("subdir/file.txt", "hello")',
            '  rctx.file("subdir/empty_dir/.keep")',
            '  rctx.delete("subdir/empty_dir/.keep")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.file("BUILD", rctx.read(rctx.path(rctx.attr.build_file)))',
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"build_file": attr.label()})'
            ),
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    other_repo_dir = self.RepoDir('other')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging: fetch my_repo only, not materialized
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'subdir')))

    # Fetch other: my_repo materialized
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(other_repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'subdir/file.txt')))
    with open(os.path.join(repo_dir, 'subdir/file.txt')) as f:
      self.assertEqual(f.read(), 'hello')
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'subdir/empty_dir')))

    # Materialized repo is not refetched after a shutdown
    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(other_repo_dir, 'BUILD')))

  def testAccessFromOtherRepo_withoutRemoteCache(self):
    # Regression test for a crash when a command that doesn't configure a remote
    # cache accesses a repo that a previous cached build injected into the
    # in-memory overlay file system but didn't materialize to disk: the
    # in-memory contents can't be served, so such repos must be re-fetched
    # from scratch.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", build_file = "@my_repo//:BUILD")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.file("BUILD", rctx.read(rctx.path(rctx.attr.build_file)))',
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"build_file": attr.label()})'
            ),
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # Populate the cache: my_repo is materialized as a side effect of fetching
    # other, which reads my_repo's BUILD file.
    self.RunBazel(['build', '@other//:haha'])

    # After expunging, fetch my_repo only: it is injected from the cache but not
    # materialized to disk and kept in memory for subsequent commands.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Access my_repo from a command without a remote cache (same server). Its
    # in-memory contents can't be served, so it must be re-fetched from scratch.
    _, _, stderr = self.RunBazel(['build', '@other//:haha', '--remote_cache='])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

  def testActionInput_withoutRemoteCache(self):
    # Like testAccessFromOtherRepo_withoutRemoteCache, but the in-memory repo is
    # accessed as a source file consumed by an action (which goes through the
    # file system's getInputStream) rather than via ctx.path()/materialization.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile(
        'BUILD.bazel',
        [
            'genrule(',
            '    name = "gen",',
            '    srcs = ["@my_repo//:data.txt"],',
            '    outs = ["out.txt"],',
            '    cmd = "cp $< $@",',
            ')',
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("data.txt", "from my_repo")',
            (
                '  rctx.file("BUILD", "exports_files([\'data.txt\'])\\n'
                "filegroup(name='haha')\")"
            ),
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # Populate the cache by building the genrule, which reads my_repo's
    # data.txt.
    self.RunBazel(['build', '//:gen'])

    # After expunging, fetch my_repo without reading data.txt: it is injected
    # from the cache but not materialized to disk and kept in memory.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data.txt')))

    # Build the genrule from a command without a remote cache (same server). The
    # action input lives only in memory and can't be served, so my_repo must be
    # re-fetched from scratch.
    _, _, stderr = self.RunBazel(['build', '//:gen', '--remote_cache='])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

  def testAccessFromOtherRepo_symlink(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", build_file = "@my_repo//:BUILD")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.symlink(rctx.path(rctx.attr.build_file), "BUILD")',
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"build_file": attr.label()})'
            ),
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    other_repo_dir = self.RepoDir('other')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # After expunging: fetch my_repo only, not materialized
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Fetch other: my_repo materialized
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(other_repo_dir, 'BUILD')))

    # Materialized repo is not refetched after a shutdown
    self.RunBazel(['shutdown'])
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(other_repo_dir, 'BUILD')))

  def testUseRepoFileInBuildRule_actionUsesCache(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'data.txt\'])")',
            '  rctx.file("data.txt", "hello")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_data",',
            '  srcs = ["@my_repo//:data.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

    # After expunging: repo and build action cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

  def do_testUseRepoFileInBuildRule_actionDoesNotUseCache(
      self, extra_flags=None
  ):
    if extra_flags is None:
      extra_flags = []
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'data.txt\'])")',
            '  rctx.file("data.txt", "hello")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_data",',
            '  srcs = ["@my_repo//:data.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '//main:use_data'] + extra_flags)
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

    # After expunging: repo and build action cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_data'] + extra_flags)
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

  def testUseRepoFileInBuildRule_actionDoesNotUseCache(self):
    self.do_testUseRepoFileInBuildRule_actionDoesNotUseCache()

  def testUseRepoFileInBuildRule_actionDoesNotUseCache_withExplicitSandboxBase(
      self,
  ):
    tmpdir = self.ScratchDir('sandbox_base')
    self.do_testUseRepoFileInBuildRule_actionDoesNotUseCache(
        extra_flags=['--sandbox_base=' + tmpdir]
    )

  def testUseSourceDirectoryInBuildRule_actionDoesNotUseCache(self):
    # Regression test for https://github.com/bazelbuild/bazel/issues/30217.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'sysroot_dir\','
                " srcs=['sysroot'], visibility=['//visibility:public'])\")"
            ),
            '  rctx.file("sysroot/include/data.txt", "source-directory-data")',
            # Verify that empty directories are materialized correctly.
            '  rctx.file("sysroot/empty_dir/.keep")',
            '  rctx.delete("sysroot/empty_dir/.keep")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "read_source_directory",',
            '  srcs = ["@my_repo//:sysroot_dir"],',
            '  outs = ["out.txt"],',
            (
                '  cmd = "cat $(location @my_repo//:sysroot_dir)/include/'
                'data.txt > $@",'
            ),
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    out = self.Path('bazel-bin/main/out.txt')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    with open(out) as f:
      self.assertEqual(f.read(), 'source-directory-data')

    # After expunging: the repo is a remote cache hit and is not fully
    # materialized, but the no-cache action still runs locally and must be able
    # to read the files below the source directory it depends on.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'sysroot/include/data.txt'))
    )
    self.assertTrue(os.path.isdir(os.path.join(repo_dir, 'sysroot/empty_dir')))
    with open(out) as f:
      self.assertEqual(f.read(), 'source-directory-data')

  def testUseSourceDirectoryWithSymlinksInBuildRule_actionDoesNotUseCache(
      self,
  ):
    # Regression test for https://github.com/bazelbuild/bazel/issues/30217 with
    # symlinks below the source directory that point out of it, at a file and a
    # directory that are not themselves inputs of the action.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'sysroot_dir\','
                " srcs=['sysroot'], visibility=['//visibility:public'])\")"
            ),
            '  rctx.file("shared/data.txt", "shared-data\\n")',
            '  rctx.file("shared/dir/nested.txt", "nested-data\\n")',
            '  rctx.symlink("shared/data.txt", "sysroot/link_to_file.txt")',
            '  rctx.symlink("shared/dir", "sysroot/link_to_dir")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "read_source_directory",',
            '  srcs = ["@my_repo//:sysroot_dir"],',
            '  outs = ["out.txt"],',
            (
                '  cmd = "cat $(location @my_repo//:sysroot_dir)/'
                'link_to_file.txt $(location @my_repo//:sysroot_dir)/'
                'link_to_dir/nested.txt > $@",'
            ),
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    out = self.Path('bazel-bin/main/out.txt')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    with open(out) as f:
      self.assertEqual(f.read(), 'shared-data\nnested-data\n')

    # After expunging: the repo is a remote cache hit and is not fully
    # materialized, but the no-cache action still runs locally and must be able
    # to read all files reachable through the source directory it depends on,
    # including via symlinks that point out of it.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(
        os.path.islink(os.path.join(repo_dir, 'sysroot/link_to_file.txt'))
    )
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'shared/data.txt')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'shared/dir/nested.txt'))
    )
    with open(out) as f:
      self.assertEqual(f.read(), 'shared-data\nnested-data\n')

  def testUseSourceDirectoryAndFileInputsInBuildRule_actionsDoNotUseCache(
      self,
  ):
    # A source directory input can overlap with regular source file inputs of
    # other actions (e.g. via glob). Both are materialized out of the remote
    # repo contents cache in the same build, through subtree materialization
    # and per-file prefetching respectively, and must not conflict.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'sysroot_dir\','
                " srcs=['sysroot'], visibility=['//visibility:public'])\\n"
                "exports_files(glob(['sysroot/**']))\")"
            ),
            (
                '  rctx.file("sysroot/include/data.txt",'
                ' "source-directory-data\\n")'
            ),
            '  rctx.file("sysroot/include/other.txt", "file-input-data\\n")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "read_source_directory",',
            '  srcs = ["@my_repo//:sysroot_dir"],',
            '  outs = ["dir_out.txt"],',
            (
                '  cmd = "cat $(location @my_repo//:sysroot_dir)/include/'
                'data.txt $(location @my_repo//:sysroot_dir)/include/'
                'other.txt > $@",'
            ),
            '  tags = ["no-cache"],',
            ')',
            'genrule(',
            '  name = "read_files",',
            (
                '  srcs = ["@my_repo//:sysroot/include/data.txt",'
                ' "@my_repo//:sysroot/include/other.txt"],'
            ),
            '  outs = ["files_out.txt"],',
            (
                '  cmd = "cat $(location @my_repo//:sysroot/include/data.txt)'
                ' $(location @my_repo//:sysroot/include/other.txt) > $@",'
            ),
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    dir_out = self.Path('bazel-bin/main/dir_out.txt')
    files_out = self.Path('bazel-bin/main/files_out.txt')
    expected = 'source-directory-data\nfile-input-data\n'

    # First fetch: not cached
    _, _, stderr = self.RunBazel(
        ['build', '//main:read_source_directory', '//main:read_files']
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    with open(dir_out) as f:
      self.assertEqual(f.read(), expected)
    with open(files_out) as f:
      self.assertEqual(f.read(), expected)

    # After expunging: the repo is a remote cache hit and both no-cache actions
    # run locally in the same build, materializing the same files through the
    # source directory subtree walk and per-file prefetching respectively.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(
        ['build', '//main:read_source_directory', '//main:read_files']
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'sysroot/include/data.txt'))
    )
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'sysroot/include/other.txt'))
    )
    with open(dir_out) as f:
      self.assertEqual(f.read(), expected)
    with open(files_out) as f:
      self.assertEqual(f.read(), expected)

  def testUseRepoSymlinkInBuildRule_actionDoesNotUseCache(self):
    # Regression test for https://github.com/bazelbuild/bazel/issues/29656.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'link2.txt\'])")',
            '  rctx.file("data.txt", "hello")',
            # A chain of symlinks ending at a regular file. Only link2.txt is a
            # declared input of the consuming action; link1.txt and data.txt are
            # reached purely by resolving it.
            '  rctx.symlink("data.txt", "link1.txt")',
            '  rctx.symlink("link1.txt", "link2.txt")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_link",',
            '  srcs = ["@my_repo//:link2.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    link2 = os.path.join(repo_dir, 'link2.txt')
    out = self.Path('bazel-bin/main/out.txt')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.islink(link2))
    self.assertTrue(os.path.exists(link2))
    with open(out) as f:
      self.assertEqual(f.read(), 'hello')

    # After expunging: the repo is a remote cache hit and is not fully
    # materialized, but the no-cache action still runs locally and must be able
    # to read the symlink it depends on.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.islink(link2))
    self.assertTrue(os.path.exists(link2))
    self.assertTrue(os.path.exists(out))
    with open(out) as f:
      self.assertEqual(f.read(), 'hello')

  def testSourceDirectoryWithSymlinkToDirectory_expandedExecutionLog(self):
    # Regression test for https://github.com/bazelbuild/bazel/issues/30264:
    # the expanded execution log walks directory inputs on the overlay file
    # system and used to crash Bazel on a symlink that resolves to a directory.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'dir\','
                " srcs=['pkg'], visibility=['//visibility:public'])\")"
            ),
            '  rctx.file("pkg/sub/data.txt", "hello")',
            # A symlink pointing at a sibling directory.
            '  rctx.symlink("pkg/sub", "pkg/sub_link")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_dir",',
            '  srcs = ["@my_repo//:dir"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $(location @my_repo//:dir)/sub/data.txt > $@",',
            ')',
        ],
    )

    # First build: the repo is fetched to disk and the action executes locally,
    # seeding the remote cache.
    _, _, stderr = self.RunBazel(['build', '//main:use_dir'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

    # After expunging, the repo is served from the in-memory overlay file
    # system and the action is a remote cache hit, so its inputs are never
    # staged locally. Logging the expanded execution log still walks the source
    # directory input on the overlay file system and encounters the symlink
    # that resolves to a directory.
    self.RunBazel(['clean', '--expunge'])
    exec_log = self.Path('exec_log.json')
    _, _, stderr = self.RunBazel([
        'build',
        '//main:use_dir',
        '--remote_download_outputs=minimal',
        '--execution_log_json_file=' + exec_log,
    ])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    with open(exec_log) as f:
      log = f.read()
    self.assertIn('pkg/sub/data.txt', log)
    self.assertIn('main/out.txt', log)

  def testRepoSymlinkChainMaterializationIsConsistent(self):
    # Full repo materialization (triggered by another repo accessing my_repo)
    # and lazy action-input materialization (triggered by a local action
    # consuming a single symlink from my_repo) must produce the same on-disk
    # representation for the symlink chain, rather than e.g. collapsing the
    # chain to its resolved target in one case but not the other.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            (
                'other_repo_rule ='
                ' use_repo_rule("//:other_repo.bzl", "other_repo_rule")'
            ),
            'other_repo_rule(name = "other", build_file = "@my_repo//:BUILD")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'haha\')\\n'
                "exports_files(['link2.txt'])\")"
            ),
            '  rctx.file("data.txt", "hello")',
            # A chain of symlinks ending at a regular file.
            '  rctx.symlink("data.txt", "link1.txt")',
            '  rctx.symlink("link1.txt", "link2.txt")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            # Reading my_repo's BUILD forces full materialization of my_repo.
            '  rctx.file("BUILD", rctx.read(rctx.attr.build_file))',
            # other is not reproducible, so it is always fetched and re-triggers
            # materialization of my_repo from the cache.
            '  return rctx.repo_metadata()',
            (
                'other_repo_rule = repository_rule(_other_repo_impl,'
                ' attrs={"build_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_link",',
            '  srcs = ["@my_repo//:link2.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    chain = ['link2.txt', 'link1.txt', 'data.txt']

    def snapshot():
      # Record a platform-independent structural layout: whether each path is a
      # symlink/file/absent and the contents it ultimately resolves to. The raw
      # symlink target string is intentionally not compared, as its
      # representation differs across platforms.
      layout = {}
      for name in chain:
        p = os.path.join(repo_dir, name)
        kind = (
            'symlink'
            if os.path.islink(p)
            else 'file'
            if os.path.isfile(p)
            else 'absent'
        )
        content = None
        if os.path.exists(p):
          with open(p) as f:
            content = f.read()
        layout[name] = (kind, content)
      return layout

    # Cold build: my_repo is fetched and uploaded to the remote repo contents
    # cache.
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Full materialization: my_repo is restored from the cache into the overlay
    # and then fully materialized because other accesses it.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    full_layout = snapshot()

    # Lazy action-input materialization: my_repo is restored from the cache into
    # the overlay and only the symlink consumed by the local action (and its
    # resolved target) is materialized.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    print('\n'.join(stderr))
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    action_layout = snapshot()

    # The symlink chain must be reproduced identically, not collapsed to its
    # resolved target: link2.txt and link1.txt must both remain symlinks (a
    # collapsed chain would drop the intermediate link1.txt) that resolve to the
    # file's contents.
    self.assertEqual(action_layout, full_layout)
    self.assertEqual(action_layout['link2.txt'], ('symlink', 'hello'))
    self.assertEqual(action_layout['link1.txt'], ('symlink', 'hello'))
    self.assertEqual(action_layout['data.txt'], ('file', 'hello'))
    if not self.IsWindows():
      # The exact relative symlink targets are reproduced (POSIX only, as the
      # representation is platform-dependent).
      self.assertEqual(
          os.readlink(os.path.join(repo_dir, 'link2.txt')), 'link1.txt'
      )
      self.assertEqual(
          os.readlink(os.path.join(repo_dir, 'link1.txt')), 'data.txt'
      )

  def testRepoWithSymlinkChainIntoMainRepoIsNotCached(self):
    # A repo containing a symlink chain that ends with a symlink pointing at a
    # file in the main repo must not be added to the remote repo contents
    # cache: when restored from the cache, the chain would only exist in the
    # in-memory overlay file system, which can't resolve symlinks that cross
    # over into the native file system.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo", main_file = "//:main_data.txt")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('main_data.txt', ['main_hello'])
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'link2.txt\'])")',
            # A chain of symlinks whose last link points (via an absolute path)
            # at a source file in the main repo.
            '  rctx.symlink(rctx.attr.main_file, "main_link.txt")',
            '  rctx.symlink("main_link.txt", "link1.txt")',
            '  rctx.symlink("link1.txt", "link2.txt")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            (
                'repo = repository_rule(_repo_impl,'
                ' attrs={"main_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_link",',
            '  srcs = ["@my_repo//:link2.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    out = self.Path('bazel-bin/main/out.txt')

    def assert_chain_on_disk():
      for name in ['link2.txt', 'link1.txt', 'main_link.txt']:
        self.assertTrue(os.path.islink(os.path.join(repo_dir, name)))
        with open(os.path.join(repo_dir, name)) as f:
          self.assertEqual(f.read(), 'main_hello\n')
      if not self.IsWindows():
        # The exact symlink targets are preserved (POSIX only, as the
        # representation is platform-dependent). The absolute symlink into the
        # main repo is not replanted to a relative path.
        self.assertEqual(
            os.readlink(os.path.join(repo_dir, 'link2.txt')), 'link1.txt'
        )
        self.assertEqual(
            os.readlink(os.path.join(repo_dir, 'link1.txt')), 'main_link.txt'
        )
        self.assertTrue(
            os.path.isabs(os.readlink(os.path.join(repo_dir, 'main_link.txt')))
        )

    # Cold build: my_repo is fetched, but not uploaded to the remote repo
    # contents cache due to the symlink pointing into the main repo.
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    assert_chain_on_disk()
    with open(out) as f:
      self.assertEqual(f.read(), 'main_hello\n')

    # After expunging, the repo is fetched again rather than restored from the
    # cache and the symlink chain keeps working.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    assert_chain_on_disk()
    with open(out) as f:
      self.assertEqual(f.read(), 'main_hello\n')

  def testRepoExternalSymlinkMaterializationIsConsistent(self):
    # A repo source symlink whose target lives in *another* repo (an absolute
    # symlink into that repo) must be reproduced identically by full repo
    # materialization and by lazy action-input materialization.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'dep_repo_rule = use_repo_rule("//:dep_repo.bzl", "dep_repo_rule")',
            'dep_repo_rule(name = "dep_repo")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            (
                'repo(name = "my_repo", external_file ='
                ' "@dep_repo//:dep_data.txt")'
            ),
            (
                'other_repo_rule ='
                ' use_repo_rule("//:other_repo.bzl", "other_repo_rule")'
            ),
            (
                'other_repo_rule(name = "other", build_file ='
                ' "@my_repo//:BUILD", dep_file = "@dep_repo//:dep_data.txt")'
            ),
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'dep_repo.bzl',
        [
            'def _dep_repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'dep_data.txt\'])")',
            '  rctx.file("dep_data.txt", "dep_hello")',
            '  return rctx.repo_metadata(reproducible=True)',
            'dep_repo_rule = repository_rule(_dep_repo_impl)',
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'haha\')\\n'
                "exports_files(['external_link.txt'])\")"
            ),
            # An absolute symlink pointing at a file in another repo.
            '  rctx.symlink(rctx.attr.external_file, "external_link.txt")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            (
                'repo = repository_rule(_repo_impl,'
                ' attrs={"external_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            # Materialize dep_repo before my_repo so that the external symlink
            # target exists when my_repo is materialized.
            '  rctx.watch(rctx.attr.dep_file)',
            '  rctx.file("BUILD", rctx.read(rctx.attr.build_file))',
            '  return rctx.repo_metadata()',
            (
                'other_repo_rule = repository_rule(_other_repo_impl,'
                ' attrs={"build_file": attr.label(), "dep_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_link",',
            '  srcs = ["@my_repo//:external_link.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    external_link = os.path.join(repo_dir, 'external_link.txt')

    def snapshot():
      # The raw (absolute) symlink target string is intentionally not compared,
      # as its representation differs across platforms; comparing whether it is
      # a symlink and what it resolves to is sufficient and portable.
      layout = {}
      layout['type'] = (
          'symlink'
          if os.path.islink(external_link)
          else 'file'
          if os.path.isfile(external_link)
          else 'absent'
      )
      # The resolved content (follows the cross-repo symlink on disk).
      layout['resolves_to'] = (
          open(external_link).read() if os.path.exists(external_link) else None
      )
      return layout

    # Cold build: my_repo and dep_repo are fetched and uploaded to the cache.
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Full materialization: my_repo is restored from the cache into the overlay
    # and fully materialized (along with dep_repo) because other accesses it.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    else:
      self.assertIn('JUST FETCHED', '\n'.join(stderr))
    full_layout = snapshot()

    # Lazy action-input materialization: only external_link.txt and its resolved
    # target in dep_repo are materialized.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_link'])
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    else:
      self.assertIn('JUST FETCHED', '\n'.join(stderr))
    action_layout = snapshot()

    # The cross-repo symlink must be reproduced (as an absolute symlink into the
    # other repo) and resolve to the other repo's file contents in both cases.
    self.assertEqual(action_layout, full_layout)
    self.assertEqual(action_layout['type'], 'symlink')
    self.assertEqual(action_layout['resolves_to'], 'dep_hello')

  def setUpExternalSymlinkWithNativeTargetRepo(self):
    """Creates a repo with a symlink pointing to another, remotely cached repo.

    The target repo is materialized on disk, while the repo containing the
    symlink is restored into the in-memory overlay if
    CROSS_REPO_SYMLINKS_CACHEABLE, and refetched otherwise.

    Returns:
      A tuple of the path of the repo containing the symlink, the path of the
      repo containing its target, and the path of the symlink.
    """
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'dep_repo_rule = use_repo_rule("//:dep_repo.bzl", "dep_repo_rule")',
            'dep_repo_rule(name = "dep_repo")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            (
                'repo(name = "my_repo", external_file ='
                ' "@dep_repo//:dep_data.txt")'
            ),
            (
                'materializer_rule ='
                ' use_repo_rule("//:materializer.bzl", "materializer_rule")'
            ),
            (
                'materializer_rule(name = "materializer", dep_file ='
                ' "@dep_repo//:dep_data.txt")'
            ),
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'dep_repo.bzl',
        [
            'def _dep_repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'dep_data.txt\'])")',
            '  rctx.file("dep_data.txt", "dep_hello")',
            '  print("JUST FETCHED DEP_REPO")',
            '  return rctx.repo_metadata(reproducible=True)',
            'dep_repo_rule = repository_rule(_dep_repo_impl)',
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'haha\')\\n'
                "exports_files(['external_link.txt'])\")"
            ),
            '  rctx.symlink(rctx.attr.external_file, "external_link.txt")',
            '  print("JUST FETCHED MY_REPO")',
            '  return rctx.repo_metadata(reproducible=True)',
            (
                'repo = repository_rule(_repo_impl,'
                ' attrs={"external_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'materializer.bzl',
        [
            'def _materializer_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  rctx.read(rctx.attr.dep_file)',
            '  return rctx.repo_metadata()',
            (
                'materializer_rule = repository_rule(_materializer_impl,'
                ' attrs={"dep_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "remote",',
            '  srcs = ["@my_repo//:external_link.txt"],',
            '  outs = ["remote.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
            'genrule(',
            '  name = "local",',
            '  srcs = ["@my_repo//:external_link.txt"],',
            '  outs = ["local.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    # Populate the remote repo contents cache without creating an action-cache
    # entry for either genrule.
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    stderr_text = '\n'.join(stderr)
    self.assertIn('JUST FETCHED DEP_REPO', stderr_text)
    self.assertIn('JUST FETCHED MY_REPO', stderr_text)

    my_repo_dir = self.RepoDir('my_repo')
    dep_repo_dir = self.RepoDir('dep_repo')
    external_link = os.path.join(my_repo_dir, 'external_link.txt')

    # Materialize only dep_repo in a fresh output base.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@materializer//:haha'])
    self.assertNotIn('JUST FETCHED DEP_REPO', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(dep_repo_dir, 'dep_data.txt')))
    self.assertFalse(os.path.lexists(external_link))

    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      # Restore my_repo into the in-memory overlay while leaving dep_repo on
      # the native file system.
      self.assertNotIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
      self.assertFalse(os.path.exists(os.path.join(my_repo_dir, 'BUILD')))
      self.assertFalse(os.path.lexists(external_link))
    else:
      # my_repo is refetched rather than restored from the cache as its symlink
      # points out of the repo.
      self.assertIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
      self.assertTrue(os.path.exists(os.path.join(my_repo_dir, 'BUILD')))
      self.assertTrue(os.path.islink(external_link))

    return my_repo_dir, dep_repo_dir, external_link

  def testRepoExternalSymlinkWithNativeTargetRepo(self):
    my_repo_dir, _, external_link = (
        self.setUpExternalSymlinkWithNativeTargetRepo()
    )

    # A remote action must be able to consume the symlink as an action input
    # without materializing or refetching its repo.
    _, _, stderr = self.RunBazel([
        'build',
        '//main:remote',
        '--spawn_strategy=remote',
        '--remote_executor=grpc://localhost:' + str(self._worker_port),
    ])
    self.assertNotIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
    with open(self.Path('bazel-bin/main/remote.txt')) as f:
      self.assertEqual(f.read(), 'dep_hello')
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertFalse(os.path.exists(os.path.join(my_repo_dir, 'BUILD')))
      self.assertFalse(os.path.lexists(external_link))

  def testRepoExternalSymlinkWithNativeTargetRepoUpload(self):
    my_repo_dir, _, external_link = (
        self.setUpExternalSymlinkWithNativeTargetRepo()
    )

    # Remove the target contents from the CAS so remote input upload has to
    # read through the symlink instead of reusing the repository blob.
    dep_blob = self.DeleteCasEntry(b'dep_hello')

    # A remote action must be able to upload the native target through the
    # symlink path.
    _, _, stderr = self.RunBazel([
        'build',
        '//main:remote',
        '--spawn_strategy=remote',
        '--remote_executor=grpc://localhost:' + str(self._worker_port),
    ])
    self.assertNotIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
    self.assertTrue(os.path.exists(dep_blob))
    with open(self.Path('bazel-bin/main/remote.txt')) as f:
      self.assertEqual(f.read(), 'dep_hello')
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertFalse(os.path.exists(os.path.join(my_repo_dir, 'BUILD')))
      self.assertFalse(os.path.lexists(external_link))

  def testRepoExternalSymlinkWithNativeTargetRepoLocalAction(self):
    my_repo_dir, _, external_link = (
        self.setUpExternalSymlinkWithNativeTargetRepo()
    )

    # A local action needs the symlink on the host file system.
    self.RunBazel([
        'build',
        '//main:local',
        '--spawn_strategy=local',
    ])
    with open(self.Path('bazel-bin/main/local.txt')) as f:
      self.assertEqual(f.read(), 'dep_hello')
    self.assertTrue(os.path.islink(external_link))
    with open(external_link) as f:
      self.assertEqual(f.read(), 'dep_hello')
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertFalse(os.path.exists(os.path.join(my_repo_dir, 'BUILD')))

  def testLostRemoteFile_build(self):
    self._runLostRemoteFileBuildTest(
        '--experimental_merged_skyframe_analysis_execution'
    )

  def testLostRemoteFile_build_noSkymeld(self):
    self._runLostRemoteFileBuildTest(
        '--noexperimental_merged_skyframe_analysis_execution'
    )

  def _runLostRemoteFileBuildTest(self, skymeld_flag):
    # Create a repo with two BUILD files (one in a subpackage), build a target
    # from one to cause it to be cached, then build that target again after
    # expunging to verify it is cached.
    # Then, restart the worker and build a target in the other build file.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )

    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'root\','
                " srcs=['root.txt'])\")"
            ),
            '  rctx.file("root.txt", "root")',
            (
                '  rctx.file("sub/BUILD", "filegroup(name=\'sub\','
                " srcs=['sub.txt'])\")"
            ),
            '  rctx.file("sub/sub.txt", "sub")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', skymeld_flag, '@my_repo//:root'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', skymeld_flag, '@my_repo//:root'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # Lose all remote files.
    self.ClearRemoteCache()

    # Build the other target: fails due to the lost input
    # TODO: #26450 - Assert success and enable the checks below.
    _, _, stderr = self.RunBazel(
        ['build', skymeld_flag, '@my_repo//sub:sub'], allow_failure=True
    )
    self.assertEqual(
        1,
        stderr.count(
            'Found transient remote cache error, retrying the build...'
        ),
    )
    canonical_repo_name = repo_dir[repo_dir.rfind('/') + 1 :]
    stderr = '\n'.join(stderr)
    self.assertRegex(
        stderr,
        'external/%s/sub/BUILD with digest .*/.* no longer available in the'
        ' remote cache'
        % re.escape(canonical_repo_name),
    )
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # After expunging again: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', skymeld_flag, '@my_repo//sub:sub'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

  def doTestMaterializationWithInternalAndExternalSymlinks(
      self, *, expect_symlinks, watch_dep_file=True
  ):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'dep_repo_rule = use_repo_rule("//:dep_repo.bzl", "dep_repo_rule")',
            'dep_repo_rule(name = "dep_repo")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            (
                'repo(name = "my_repo",'
                ' external_file = "@dep_repo//:dep_data.txt")'
            ),
            (
                'other_repo_rule ='
                ' use_repo_rule("//:other_repo.bzl", "other_repo_rule")'
            ),
            (
                'other_repo_rule(name = "other",'
                ' build_file = "@my_repo//:BUILD",'
                ' dep_file = "@dep_repo//:dep_data.txt")'
            ),
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'dep_repo.bzl',
        [
            'def _dep_repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'dep_data.txt\'])")',
            '  rctx.file("dep_data.txt", "dep_hello")',
            '  print("JUST FETCHED DEP_REPO")',
            '  return rctx.repo_metadata(reproducible=True)',
            'dep_repo_rule = repository_rule(_dep_repo_impl)',
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            '  rctx.file("data.txt", "hello")',
            '  rctx.symlink("data.txt", "internal_link.txt")',
            '  rctx.symlink(rctx.attr.external_file, "external_link.txt")',
            '  print("JUST FETCHED MY_REPO")',
            '  return rctx.repo_metadata(reproducible=True)',
            (
                'repo = repository_rule(_repo_impl,'
                ' attrs={"external_file": attr.label()})'
            ),
        ],
    )
    other_repo_lines = [
        'def _other_repo_impl(rctx):',
    ]
    if watch_dep_file:
      # Materialize dep_repo before my_repo so that the external
      # symlink target exists when my_repo is materialized.
      other_repo_lines.append('  rctx.watch(rctx.attr.dep_file)')
    other_repo_lines.extend([
        '  rctx.file("BUILD", rctx.read(rctx.attr.build_file))',
        # other_repo is not reproducible, so it is always fetched
        # and triggers materialization of my_repo.
        '  return rctx.repo_metadata()',
        (
            'other_repo_rule = repository_rule(_other_repo_impl,'
            ' attrs={"build_file": attr.label(),'
            ' "dep_file": attr.label()})'
        ),
    ])
    self.ScratchFile('other_repo.bzl', other_repo_lines)

    repo_dir = self.RepoDir('my_repo')
    internal_link = os.path.join(repo_dir, 'internal_link.txt')
    external_link = os.path.join(repo_dir, 'external_link.txt')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    stderr_text = '\n'.join(stderr)
    self.assertIn('JUST FETCHED MY_REPO', stderr_text)
    self.assertIn('JUST FETCHED DEP_REPO', stderr_text)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    if expect_symlinks:
      self.assertTrue(os.path.islink(internal_link))
      self.assertTrue(os.path.islink(external_link))
    with open(internal_link) as f:
      self.assertEqual(f.read(), 'hello')
    with open(external_link) as f:
      self.assertEqual(f.read(), 'dep_hello')

    # After expunging: my_repo cached, not materialized
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    if CROSS_REPO_SYMLINKS_CACHEABLE or not expect_symlinks:
      # Without symlink support, rctx.symlink falls back to copying the target
      # file, so the repo doesn't contain any symlinks pointing out of it and
      # is cached even when repos with cross-repo symlinks aren't.
      self.assertNotIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
      self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    else:
      # my_repo is refetched, which also materializes dep_repo as its file is
      # referenced by label.
      self.assertIn('JUST FETCHED MY_REPO', '\n'.join(stderr))
      self.assertNotIn('JUST FETCHED DEP_REPO', '\n'.join(stderr))
      self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

    # Fetch other: my_repo materialized; dep_repo only if watch_dep_file.
    _, _, stderr = self.RunBazel(['build', '@other//:haha'])
    stderr_text = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED MY_REPO', stderr_text)
    self.assertNotIn('JUST FETCHED DEP_REPO', stderr_text)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    if expect_symlinks:
      self.assertTrue(os.path.islink(internal_link))
      self.assertTrue(os.path.islink(external_link))
    with open(internal_link) as f:
      self.assertEqual(f.read(), 'hello')
    if watch_dep_file or not CROSS_REPO_SYMLINKS_CACHEABLE:
      with open(external_link) as f:
        self.assertEqual(f.read(), 'dep_hello')
    else:
      # dep_repo was not materialized, so the external symlink is dangling.
      self.assertFalse(os.path.exists(external_link))

  def testMaterializationWithInternalAndExternalSymlinks(self):
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          [
              'startup --windows_enable_symlinks',
          ],
          mode='a',
      )
    self.doTestMaterializationWithInternalAndExternalSymlinks(
        expect_symlinks=True
    )

  def testMaterializationWithInternalAndExternalSymlinks_withoutSymlinksOnWindows(
      self,
  ):
    if not self.IsWindows():
      self.skipTest('This test is only relevant on Windows')
    self.doTestMaterializationWithInternalAndExternalSymlinks(
        expect_symlinks=False
    )

  def testMaterializationWithDanglingExternalSymlink(self):
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          [
              'startup --windows_enable_symlinks',
          ],
          mode='a',
      )
    self.doTestMaterializationWithInternalAndExternalSymlinks(
        expect_symlinks=True, watch_dep_file=False
    )

  def testBzlFilePrefetching(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", """',
            'load(":nested.bzl", "nested_fg")',
            'nested_fg(name = "haha")',
            '""")',
            '  rctx.file("nested.bzl", """',
            'load("//subdir:more_nested.bzl", "more_nested_fg")',
            'def nested_fg(name):',
            '  more_nested_fg(name = name)',
            '""")',
            '  rctx.file("subdir/BUILD")',
            '  rctx.file("subdir/more_nested.bzl", """',
            'def more_nested_fg(name):',
            '  native.filegroup(name = name)',
            '""")',
            '  rctx.file("file.txt", "hello")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'file.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'subdir/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'nested.bzl')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'subdir/more_nested.bzl'))
    )

    # After expunging: cached, .bzl files materialized
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'file.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'subdir/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'nested.bzl')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'subdir/more_nested.bzl'))
    )

    # After expunging, without using repo contents cache: not cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel([
        '--noexperimental_remote_repo_contents_cache',
        'build',
        '@my_repo//:haha',
    ])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'file.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'subdir/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'nested.bzl')))
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'subdir/more_nested.bzl'))
    )

  def testBzlSymlinkLoadedByBuildFile(self):
    # Regression test for
    # https://github.com/bazelbuild/bazel/issues/29656#issuecomment-4808145049.
    #
    # A repo's BUILD file loads a .bzl file that is a symlink. On a remote repo
    # contents cache hit, the repo is injected into the overlay file system but
    # not materialized on disk. Reads of .bzl (and REPO.bazel) files are
    # redirected to the native file system on the assumption that they were
    # prefetched during injection, but symlinks are not prefetched, only their
    # target if they match the name pattern.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", """',
            'load(":helper.bzl", "the_name")',
            'filegroup(name = the_name)',
            '""")',
            '  rctx.file("real_helper.bzl", \'the_name = "haha"\')',
            '  rctx.symlink("real_helper.bzl", "helper.bzl")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.islink(os.path.join(repo_dir, 'helper.bzl')))

    # After expunging: cached. The repo is injected but not materialized; the
    # symlinked .bzl file loaded by the BUILD file must still be readable.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'helper.bzl')))

  def testBzlSymlinkToOtherRepoLoadedByBuildFile(self):
    # Regression test for
    # https://github.com/bazelbuild/bazel/issues/29656#issuecomment-4808145049.
    #
    # Cross-repo variant of testBzlSymlinkLoadedByBuildFile.
    if self.IsWindows():
      self.ScratchFile(
          '.bazelrc',
          ['startup --windows_enable_symlinks'],
          mode='a',
      )
    self.ScratchFile(
        'MODULE.bazel',
        [
            'helper_repo = use_repo_rule("//:helper_repo.bzl", "helper_repo")',
            'helper_repo(name = "helper_repo")',
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'helper_repo.bzl',
        [
            'def _helper_repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'helper.bzl\'])")',
            '  rctx.file("helper.bzl", \'the_name = "haha"\')',
            '  print("JUST FETCHED HELPER")',
            '  return rctx.repo_metadata(reproducible=True)',
            'helper_repo = repository_rule(_helper_repo_impl)',
        ],
    )
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", """',
            'load(":helper.bzl", "the_name")',
            'filegroup(name = the_name)',
            '""")',
            '  rctx.symlink(Label("@helper_repo//:helper.bzl"), "helper.bzl")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.islink(os.path.join(repo_dir, 'helper.bzl')))

    # After expunging: cached. my_repo is injected but not materialized; the
    # cross-repo symlinked .bzl loaded by the BUILD file must still be readable.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    if CROSS_REPO_SYMLINKS_CACHEABLE:
      self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
      self.assertFalse(os.path.exists(os.path.join(repo_dir, 'helper.bzl')))
    else:
      # my_repo is refetched, which also materializes helper_repo as its file
      # is referenced by label.
      self.assertIn('JUST FETCHED', '\n'.join(stderr))
      self.assertNotIn('JUST FETCHED HELPER', '\n'.join(stderr))
      self.assertTrue(os.path.islink(os.path.join(repo_dir, 'helper.bzl')))

  def testRun(self):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'bazel_dep(name = "buildozer", version = "8.5.1")',
        ],
    )

    # First fetch: not cached
    _, stdout, _ = self.RunBazel(['run', '@buildozer', '--', '--version'])
    self.assertIn('buildozer version: 8.5.1', '\n'.join(stdout))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, stdout, _ = self.RunBazel(['run', '@buildozer', '--', '--version'])
    self.assertIn('buildozer version: 8.5.1', '\n'.join(stdout))
    repo_dir = self.RepoDir('buildozer')
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'MODULE.bazel')))

  def testReverseDependencyDirection(self):
    # Set up two repos that retain their predeclared input hashes across two
    # builds but still reverse their dependency direction. Depending on how repo
    # cache candidates are checked, this could lead to a Skyframe cycle.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(',
            '  name = "foo",',
            '  deps_file = "//:foo_deps.txt",',
            ')',
            'repo(',
            '  name = "bar",',
            '  deps_file = "//:bar_deps.txt",',
            ')',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  deps = rctx.read(rctx.attr.deps_file).splitlines()',
            '  output = ""',
            '  for dep in deps:',
            '    if dep:',
            '      output += "{}: {}\\n".format(dep, rctx.read(Label(dep)))',
            '  rctx.file("output.txt", output)',
            '  rctx.file("BUILD", "exports_files([\'output.txt\'])")',
            '  print("JUST FETCHED: %s" % rctx.original_name)',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(',
            '  implementation = _repo_impl,',
            '  attrs = {',
            '    "deps_file": attr.label(),  }',
            ')',
        ],
    )

    self.ScratchFile('foo_deps.txt', ['@bar//:output.txt'])
    self.ScratchFile('bar_deps.txt', [''])

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@foo//:output.txt'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED: bar', stderr)
    self.assertIn('JUST FETCHED: foo', stderr)

    # After expunging and reversing the dependency direction: not cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('foo_deps.txt', [''])
    self.ScratchFile('bar_deps.txt', ['@foo//:output.txt'])
    _, _, stderr = self.RunBazel(['build', '@foo//:output.txt'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED: foo', stderr)
    self.assertNotIn('JUST FETCHED: bar', stderr)

    # After expunging and reversing the dependency direction: both cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('foo_deps.txt', ['@bar//:output.txt'])
    self.ScratchFile('bar_deps.txt', [''])
    _, _, stderr = self.RunBazel(['build', '@foo//:output.txt'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)

  def testMaterializedRepoIsNotRefetchedWhenEvictedFromCache(self):
    # A repo restored from the remote repo contents cache is injected into an
    # in-memory overlay file system. When a file is later materialized on disk,
    # a contents proxy is recorded on its injected metadata. The proxy lets the
    # next build recognize the materialized file (which has no fast digest on
    # the local file system) as unchanged; without it, the file is compared by
    # contents proxy rather than digest, is considered modified, and the repo
    # is spuriously invalidated. To make such an invalidation observable,
    # the remote cache is evicted first, so a refetch can no longer be served
    # silently from the cache and must visibly re-run the repo rule.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", data_file = "@my_repo//:data.txt")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "exports_files([\'data.txt\'])")',
            '  rctx.file("data.txt", "hello")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            # Reading my_repo's data.txt forces full materialization of
            # my_repo, recording a contents proxy on each materialized file.
            # other is not reproducible, so it is always refetched
            # and re-triggers materialization.
            '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
            (
                '  rctx.file("data_copy.txt",'
                ' rctx.read(rctx.path(rctx.attr.data_file)))'
            ),
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"data_file": attr.label()})'
            ),
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_data",',
            '  srcs = ["@my_repo//:data.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $< > $@",',
            '  tags = ["no-cache"],',
            ')',
        ],
    )

    # Cold build: fetch my_repo and upload it to the remote repo contents cache.
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Restore my_repo from the cache into the overlay; the genrule reads
    # data.txt while it is still
    # overlay-resident, so its metadata is the injected remote metadata.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # Fully materialize my_repo onto the local disk (other reads its files),
    # which records a contents
    # proxy on the injected metadata and evicts the repo from the overlay.
    self.RunBazel(['build', '@other//:haha'])

    # Evict everything from the remote cache. After this, my_repo can no longer
    # be silently restored
    # from the cache: any refetch must visibly re-run its repo rule.
    self.ClearRemoteCache()

    # Incremental build: the materialized repo file is recognized as unchanged
    # via its contents proxy, so my_repo is neither invalidated nor refetched.
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))


if __name__ == '__main__':
  absltest.main()
