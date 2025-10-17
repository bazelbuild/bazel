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

import json
import os
import re
import tempfile
import time
from absl.testing import absltest

from src.test.py.bazel import test_base


class RemoteRepoContentsCacheTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    worker_port = self.StartRemoteWorker()
    self.ScratchFile(
      '.bazelrc',
      [
        'startup --experimental_remote_repo_contents_cache',
        # Only use the remote repo contents cache.
        'common --repo_contents_cache=',
        'common --remote_cache=grpc://localhost:' + str(worker_port),
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
    _, _, stderr = self.RunBazel(
      ['--noexperimental_remote_repo_contents_cache', 'build', '@my_repo//:haha']
    )
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

    # Change back to previous recorded inputs: not cached
    # TODO: This is the current behavior, but it's not desired. Support for
    #  caching repos with dynamic deps should be added.
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))

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
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_a, env_add={'LOLOL': 'lol'})
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir_a, 'BUILD')))

    # Fetch in B (with same env variable): cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b, env_add={'LOLOL': 'lol'})
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir_b, 'BUILD')))

    # Change env variable: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b, env_add={'LOLOL': 'rofl'})
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir_b, 'BUILD')))

    # Building A again even after expunging: cached
    self.RunBazel(['clean', '--expunge'], cwd=dir_a)
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_a, env_add={'LOLOL': 'rofl'})
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
        'other_repo = repository_rule(_other_repo_impl, attrs={"build_file": attr.label()})',
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
        'other_repo = repository_rule(_other_repo_impl, attrs={"build_file": attr.label()})',
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

  def testUseRepoFileInBuildRule_actionDoesNotUseCache(self):
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
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

    # After expunging: repo and build action cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_data'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    self.assertTrue(os.path.exists(self.Path('bazel-bin/main/out.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'hello')

  def testLostRemoteFile_build(self):
    # Create a repo with two BUILD files (one in a subpackage), build a target from one to cause it to be cached, then build that target again after expunging to verify it is cached.
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
        '  rctx.file("BUILD", "filegroup(name=\'root\', srcs=[\'root.txt\'])")',
        '  rctx.file("root.txt", "root")',
        '  rctx.file("sub/BUILD", "filegroup(name=\'sub\', srcs=[\'sub.txt\'])")',
        '  rctx.file("sub/sub.txt", "sub")',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl)',
      ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:root'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:root'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # Lose all remote files.
    self.ClearRemoteCache()

    # Build the other target: fails due to the lost input
    # TODO: Assert success and enable the checks below after fixing #26450.
    _, _, stderr = self.RunBazel(['build', '@my_repo//sub:sub'], allow_failure=True)
    self.assertEqual(1, stderr.count('Found transient remote cache error, retrying the build...'))
    canonical_repo_name = repo_dir[repo_dir.rfind('/') + 1:]
    stderr = '\n'.join(stderr)
    self.assertRegex(stderr,
                     'external/%s/sub/BUILD with digest .*/.* no longer available in the remote cache' % re.escape(
                       canonical_repo_name))
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # After expunging again: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//sub:sub'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))


if __name__ == '__main__':
  absltest.main()
