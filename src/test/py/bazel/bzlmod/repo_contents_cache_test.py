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

import os
import tempfile
import time

from absl.testing import absltest
from src.test.py.bazel import test_base


class RepoContentsCacheTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.repo_contents_cache = tempfile.mkdtemp(dir=self._tests_root).replace(
        '\\', '/'
    )
    self.ScratchFile(
        '.bazelrc',
        [
            'build --verbose_failures',
            'common --repo_contents_cache=%s' % self.repo_contents_cache,
        ],
    )

  def hasCacheEntry(self):
    for l1 in os.listdir(self.repo_contents_cache):
      l1_path = os.path.join(self.repo_contents_cache, l1)
      if l1 != '_trash' and os.path.isdir(l1_path):
        for l2 in os.listdir(l1_path):
          if l2.endswith('.recorded_inputs'):
            # we still have some cache entries!
            return True
    return False

  def sleepUntilCacheEmpty(self):
    for _ in range(10):
      if not self.hasCacheEntry():
        return
      time.sleep(0.5)
    self.fail('repo contents cache still not empty after 5 seconds')

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
    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging, without using repo contents cache: not cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(
        ['build', '--repo_contents_cache=', '@my_repo//:haha']
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

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

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

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

  def testNotCachedWhenRecordedInputsChange(self):
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

    # First fetch: not cached
    self.ScratchFile('data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Change recorded inputs: not cached
    self.ScratchFile('data.txt', ['two'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Change back to previous recorded inputs: cached (even after expunging)
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

  def testNotCachedWhenRecordedInputsChange_envVar(self):
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
            '  rctx.getenv("LOLOL")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    # First fetch: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'lol'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Change recorded inputs: not cached
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'kek'}
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Change back to previous recorded inputs: cached (even after expunging)
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(
        ['build', '@my_repo//:haha'], env_add={'LOLOL': 'lol'}
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

  def testNoThrashingBetweenWorkspaces(self):
    module_bazel_lines = [
        'repo = use_repo_rule("//:repo.bzl", "repo")',
        'repo(name = "my_repo")',
    ]
    repo_bzl_lines = [
        'def _repo_impl(rctx):',
        '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
        '  rctx.watch(Label("@//:data.txt"))',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl)',
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

    # First fetch in A: not cached
    self.ScratchFile('a/data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_a)
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Fetch in B (with same 'data.txt'): cached
    self.ScratchFile('b/data.txt', ['one'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b)
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # Change 'b/data.txt': not cached
    self.ScratchFile('b/data.txt', ['two'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b)
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Building A again even after expunging: cached
    self.RunBazel(['clean', '--expunge'], cwd=dir_a)
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_a)
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

  def testGc_singleServer_gcAfterCacheHit(self):
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
    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging, but with a very quick GC delay & max age: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel([
        'build',
        '--repo_contents_cache_gc_max_age=1ms',
        '--repo_contents_cache_gc_idle_delay=1ms',
        '@my_repo//:haha',
    ])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # GC'd while server is alive: not cached, but also no crash
    self.sleepUntilCacheEmpty()
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

  def testGc_singleServer_gcAfterCacheMiss(self):
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
    # First fetch: not cached
    _, _, stderr = self.RunBazel([
        'build',
        '--repo_contents_cache_gc_max_age=1ms',
        '--repo_contents_cache_gc_idle_delay=1ms',
        '@my_repo//:haha',
    ])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # GC'd while server is alive: not cached, but also no crash
    self.sleepUntilCacheEmpty()
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

  def testGc_multipleServers(self):
    module_bazel_lines = [
        'repo = use_repo_rule("//:repo.bzl", "repo")',
        'repo(name = "my_repo")',
    ]
    repo_bzl_lines = [
        'def _repo_impl(rctx):',
        '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl)',
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

    # First fetch in A: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_a)
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Fetch in B with subsequent GC run: cached
    _, _, stderr = self.RunBazel(
        [
            'build',
            '--repo_contents_cache_gc_max_age=1ms',
            '--repo_contents_cache_gc_idle_delay=1ms',
            '@my_repo//:haha',
        ],
        cwd=dir_b,
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # GC'd while A's server is alive (after A's earlier cache miss) with
    # subsequent GC run: not cached, but also no crash
    self.sleepUntilCacheEmpty()
    _, _, stderr = self.RunBazel(
        [
            'build',
            '--repo_contents_cache_gc_max_age=1ms',
            '--repo_contents_cache_gc_idle_delay=1ms',
            '@my_repo//:haha',
        ],
        cwd=dir_a,
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # GC'd while B's server is alive (after B's earlier cache hit):
    # not cached, but also no crash
    self.sleepUntilCacheEmpty()
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b)
    self.assertIn('JUST FETCHED', '\n'.join(stderr))


if __name__ == '__main__':
  absltest.main()
