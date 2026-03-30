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
# pylint: disable=g-bad-todo
# pylint: disable=g-long-ternary

import json
import os
import pathlib
import shutil
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

  def repoDir(self, repo_name, cwd=None):
    _, stdout, _ = self.RunBazel(['info', 'output_base'], cwd=cwd)
    self.assertLen(stdout, 1)
    output_base = stdout[0].strip()

    _, stdout, _ = self.RunBazel(['mod', 'dump_repo_mapping', ''], cwd=cwd)
    self.assertLen(stdout, 1)
    mapping = json.loads(stdout[0])
    canonical_repo_name = mapping[repo_name]

    return output_base + '/external/' + canonical_repo_name

  def assertRepoCached(self, repo_dir):
    """Assert that a repo dir is a symlink into the repo contents cache."""
    try:
      target_path = os.readlink(repo_dir)
      real_target_path = os.path.realpath(target_path)
      real_repo_contents_cache = os.path.realpath(self.repo_contents_cache)
      for parent in pathlib.Path(real_target_path).parents:
        if parent.samefile(real_repo_contents_cache):
          return
      self.fail(
          'repo target dir %s is not in the repo contents cache %s'
          % (real_target_path, real_repo_contents_cache)
      )
    except OSError:
      self.fail('repo_dir %s is not a symlink or junction' % repo_dir)

  def assertRepoNotCached(self, repo_dir):
    """Assert that a repo dir is NOT a symlink into the repo contents cache."""
    try:
      target_path = os.readlink(repo_dir)
      real_target_path = os.path.realpath(target_path)
      real_repo_contents_cache = os.path.realpath(self.repo_contents_cache)
      for parent in pathlib.Path(real_target_path).parents:
        if parent.samefile(real_repo_contents_cache):
          self.fail(
              'repo with cross-repo symlinks should not be cached, but %s'
              ' points into %s' % (real_target_path, real_repo_contents_cache)
          )
    except OSError:
      pass  # Not a symlink means not cached, which is expected

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
    # Verify that the repo directory under the output base is a symlink or
    # junction into the repo contents cache.
    repo_dir = self.repoDir('my_repo')
    try:
      target_path = os.readlink(repo_dir)
      real_target_path = os.path.realpath(target_path)
      real_repo_contents_cache = os.path.realpath(self.repo_contents_cache)
      for parent in pathlib.Path(real_target_path).parents:
        if parent.samefile(real_repo_contents_cache):
          break
      else:
        self.fail(
            'repo target dir %s is not in the repo contents cache %s'
            % (real_target_path, real_repo_contents_cache)
        )
    except OSError:
      self.fail('repo_dir %s is not a symlink or junction' % repo_dir)

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
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)

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
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)

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
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)

    # GC'd while B's server is alive (after B's earlier cache hit):
    # not cached, but also no crash
    self.sleepUntilCacheEmpty()
    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'], cwd=dir_b)
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)

  def testReverseDependencyDirection(self):
    # Set up two repos that retain their predeclared input hashes across two
    #  builds but still reverse their dependency direction. Depending on how
    # repo cache candidates are checked, this could lead to a Skyframe cycle.
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
    self.assertIn('JUST FETCHED: bar', '\n'.join(stderr))
    self.assertIn('JUST FETCHED: foo', '\n'.join(stderr))

    # After expunging and reversing the dependency direction: not cached
    self.RunBazel(['clean', '--expunge'])
    self.ScratchFile('foo_deps.txt', [''])
    self.ScratchFile('bar_deps.txt', ['@foo//:output.txt'])
    self.RunBazel(['build', '@foo//:output.txt'])

  def doTestRepoContentsCacheDeleted(self, check_external_repository_files):
    repo_contents_cache = self.ScratchDir('repo_contents_cache')
    workspace = self.ScratchDir('workspace')
    extra_args = [
        '--experimental_check_external_repository_files=%s'
        % str(check_external_repository_files).lower(),
        '--repo_contents_cache=%s' % repo_contents_cache,
    ]

    self.ScratchFile(
        'workspace/MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
        ],
    )
    self.ScratchFile(
        'workspace/BUILD.bazel',
        [
            'genrule(',
            '  name = "gen",',
            '  srcs = ["@my_repo//:haha", "in.txt"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $(SRCS) > $(OUTS)",',
            ')',
        ],
    )
    self.ScratchFile(
        'workspace/repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD", "filegroup(name=\'haha\','
                " srcs=['a.txt'], visibility=['//visibility:public'])\")"
            ),
            '  rctx.file("a.txt", "hello world")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    # First fetch: not cached
    self.ScratchFile('workspace/in.txt', ['1'])
    _, _, stderr = self.RunBazel(
        [
            'build',
            '//:gen',
        ]
        + extra_args,
        cwd=workspace,
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    with open(os.path.join(workspace, 'bazel-bin/out.txt'), 'r') as f:
      self.assertEqual(f.read(), 'hello world1\n')

    # Second fetch: cached
    self.ScratchFile('workspace/in.txt', ['2'])
    _, _, stderr = self.RunBazel(
        [
            'build',
            '//:gen',
        ]
        + extra_args,
        cwd=workspace,
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    with open(os.path.join(workspace, 'bazel-bin/out.txt'), 'r') as f:
      self.assertEqual(f.read(), 'hello world2\n')

    # Delete the entire repo contents cache and fetch again: not cached
    # Avoid access denied on Windows due to files being read-only by moving to
    # a different location instead.
    os.rename(repo_contents_cache, repo_contents_cache + '_deleted')
    self.ScratchFile('workspace/in.txt', ['3'])
    _, _, stderr = self.RunBazel(
        ['build', '//:gen'] + extra_args,
        cwd=workspace,
    )
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)
    with open(os.path.join(workspace, 'bazel-bin/out.txt'), 'r') as f:
      self.assertEqual(f.read(), 'hello world3\n')

    # Second fetch after deletion: cached
    self.ScratchFile('workspace/in.txt', ['4'])
    _, _, stderr = self.RunBazel(
        ['build', '//:gen'] + extra_args,
        cwd=workspace,
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertNotIn('WARNING', '\n'.join(stderr))
    with open(os.path.join(workspace, 'bazel-bin/out.txt'), 'r') as f:
      self.assertEqual(f.read(), 'hello world4\n')

    # Delete the entire repo contents cache and fetch again with a different
    # path: not cached
    # Avoid access denied on Windows due to files being read-only by moving to
    # a different location instead.
    os.rename(repo_contents_cache, repo_contents_cache + '_deleted_again')
    self.ScratchFile('workspace/in.txt', ['5'])
    _, _, stderr = self.RunBazel(
        ['build', '//:gen']
        + extra_args
        + [
            '--repo_contents_cache=%s' % repo_contents_cache + '2',
        ],
        cwd=workspace,
    )
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)
    with open(os.path.join(workspace, 'bazel-bin/out.txt'), 'r') as f:
      self.assertEqual(f.read(), 'hello world5\n')

  def testRepoContentsCacheDeleted_withCheckExternalRepositoryFiles(self):
    self.doTestRepoContentsCacheDeleted(check_external_repository_files=True)

  def testRepoContentsCacheDeleted_withoutCheckExternalRepositoryFiles(self):
    self.doTestRepoContentsCacheDeleted(check_external_repository_files=False)

  def doTestCachedRepoWithSymlinks(self, expect_cross_repo_cached=False):
    self.ScratchFile(
        'MODULE.bazel',
        [
            'ext = use_extension("extension.bzl", "ext")',
            'use_repo(ext, "foo", "bar")',
        ],
    )
    # Create the external file outside the workspace so it's not treated as
    # a cross-repo symlink (workspace files go through _main redirect).
    abs_file = os.path.join(self._tests_root, 'abs_external')
    with open(abs_file, 'w') as f:
      f.write('Hello from abs!\n')
    abs_foo = abs_file.replace('\\', '/')
    self.ScratchFile(
        'extension.bzl',
        [
            'def _repo_foo_impl(ctx):',
            '    ctx.file("REPO.bazel")',
            '    ctx.file("data", "Hello from foo!\\n")',
            '    ctx.file("sub/data", "Hello from sub!\\n")',
            # Relative same-repo symlink (not touched by replanting)
            '    ctx.symlink("data", "sym_rel")',
            # Absolute same-repo symlink (replanted to relative before caching)
            '    ctx.symlink(ctx.path("data"), "sym_abs_self")',
            # Absolute same-repo symlink to a file in a subdirectory
            '    ctx.symlink(ctx.path("sub/data"), "sym_abs_sub")',
            # Absolute symlink outside of Bazel (not touched by replanting)
            f'    ctx.symlink("{abs_foo}", "sym_ext")',
            (
                '    ctx.file("BUILD", "exports_files([\'sym_rel\','
                " 'sym_abs_self', 'sym_abs_sub', 'sym_ext'])\")"
            ),
            '    return ctx.repo_metadata(reproducible=True)',
            'repo_foo = repository_rule(implementation=_repo_foo_impl)',
            '',
            'def _repo_bar_impl(ctx):',
            '    ctx.file("REPO.bazel")',
            '    ctx.file("data", "Hello from bar!\\n")',
            # Cross-repo symlink: prevents local caching on Unix
            '    ctx.symlink(ctx.path(Label("@foo//:data")), "sym_cross")',
            '    ctx.file("BUILD", "exports_files([\'data\', \'sym_cross\'])")',
            '    return ctx.repo_metadata(reproducible=True)',
            'repo_bar = repository_rule(implementation=_repo_bar_impl)',
            '',
            'def _ext_impl(ctx):',
            '    repo_foo(name="foo")',
            '    repo_bar(name="bar")',
            'ext = module_extension(implementation=_ext_impl)',
        ],
    )
    self.ScratchFile(
        'BUILD',
        [
            'genrule(',
            '  name = "print_paths",',
            (
                '  srcs = ["@foo//:sym_rel", "@foo//:sym_abs_self",'
                ' "@foo//:sym_abs_sub", "@foo//:sym_ext", "@bar//:sym_cross"],'
            ),
            '  outs = ["output.txt"],',
            '  cmd = "cat $(SRCS) > $@",',
            ')',
        ],
    )
    # First build: fetches and caches the repos.
    self.RunBazel(['build', '//:print_paths'])
    output = os.path.join(self._test_cwd, 'bazel-bin/output.txt')
    self.AssertFileContentContains(
        output,
        'Hello from foo!\nHello from foo!\nHello from sub!\nHello from abs!'
        '\nHello from foo!\n',
    )
    # Verify that foo (only same-repo symlinks) is cached.
    foo_dir = self.repoDir('foo')
    self.assertRepoCached(foo_dir)

    bar_dir = self.repoDir('bar')
    if expect_cross_repo_cached:
      self.assertRepoCached(bar_dir)
    else:
      self.assertRepoNotCached(bar_dir)

    # Copy the workspace to a new location and use a new output base to
    # verify that the cached same-repo symlinks are portable.
    new_wd = self._test_cwd + '/new'
    shutil.copytree(self._test_cwd, new_wd, symlinks=True)
    output_base = tempfile.mkdtemp(dir=self._tests_root)
    self.RunBazel(
        [
            f'--output_base={output_base}',
            'build',
            '//:print_paths',
            '--verbose_failures',
        ],
        cwd=new_wd,
    )
    output = os.path.join(new_wd, 'bazel-bin/output.txt')
    self.AssertFileContentContains(
        output,
        'Hello from foo!\nHello from foo!\nHello from sub!\nHello from abs!'
        '\nHello from foo!\n',
    )

  def testCachedRepoWithSymlinks(self):
    # On Windows without --windows_enable_symlinks, symlinks are just file
    # copies, so the cross-repo repo is still cached.
    # TODO: Ensure that symlinks that are created as copies are tracked for
    #  invalidation.
    self.doTestCachedRepoWithSymlinks(expect_cross_repo_cached=self.IsWindows())

  def testCachedRepoWithSymlinks_symlinksEnabledOnWindows(self):
    if not self.IsWindows():
      self.skipTest('This test is only relevant on Windows')
    self.ScratchFile(
        '.bazelrc',
        [
            'startup --windows_enable_symlinks',
        ],
        mode='a',
    )
    # With --windows_enable_symlinks, real symlinks are created and detected
    # by the replanting logic, so cross-repo symlinks prevent caching.
    self.doTestCachedRepoWithSymlinks(expect_cross_repo_cached=False)


if __name__ == '__main__':
  absltest.main()
