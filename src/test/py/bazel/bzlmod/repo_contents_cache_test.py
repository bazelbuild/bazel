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
            'common --experimental_check_external_repository_files=%s' % str(self.checkExternalRepositoryFiles()).lower(),
        ]
    )

  def checkExternalRepositoryFiles(self):
    return os.getenv('CHECK_EXTERNAL_REPOSITORY_FILES') == 'True'

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
    # Verify that the repo directory under the output base is a symlink into
    # the repo contents cache.
    repo_dir = self.repoDir('my_repo')
    self.assertTrue(os.path.islink(repo_dir))
    target_path = os.readlink(repo_dir)
    self.assertTrue(target_path.startswith(self.repo_contents_cache))

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
    exit_code, stdout, stderr = self.RunBazel(
        ['build', '@foo//:output.txt'], allow_failure=True
    )
    # TODO: b/xxxxxxx - This is NOT the intended behavior.
    self.AssertNotExitCode(exit_code, 0, stderr, stdout)
    self.assertIn('.-> @@+repo+foo', stderr)
    self.assertIn('|   @@+repo+bar', stderr)
    self.assertIn('`-- @@+repo+foo', stderr)

  def testRepoContentsCacheDeleted(self):
    repo_contents_cache = self.ScratchDir('repo_contents_cache')
    workspace = self.ScratchDir('workspace')

    self.ScratchFile(
      'workspace/MODULE.bazel',
      [
        'repo = use_repo_rule("//:repo.bzl", "repo")',
        'repo(name = "my_repo")',
      ],
    )
    self.ScratchFile('workspace/BUILD.bazel')
    self.ScratchFile(
      'workspace/repo.bzl',
      [
        'def _repo_impl(rctx):',
        '  rctx.file("BUILD", "filegroup(name=\'haha\')")',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl)',
      ],
    )
    # First fetch: not cached
    _, _, stderr = self.RunBazel(
      ['build', '@my_repo//:haha', '--repo_contents_cache=%s' % repo_contents_cache],
      cwd=workspace,
    )
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Second fetch: cached
    _, _, stderr = self.RunBazel(
      ['build', '@my_repo//:haha', '--repo_contents_cache=%s' % repo_contents_cache],
      cwd=workspace,
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))

    # Delete the entire repo contents cache and fetch again: not cached
    shutil.rmtree(repo_contents_cache)
    _, _, stderr = self.RunBazel(
      ['build', '@my_repo//:haha', '--repo_contents_cache=%s' % repo_contents_cache],
      cwd=workspace,
    )
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertNotIn('WARNING', stderr)

  def testEntryCorrupted(self):
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

    # Corrupt the only cache entry
    found = False
    for l1 in os.listdir(self.repo_contents_cache):
      l1_path = os.path.join(self.repo_contents_cache, l1)
      if os.path.isdir(l1_path) and 'my_repo' in l1:
        for l2 in os.listdir(l1_path):
          if not l2.endswith('.recorded_inputs'):
            with open(os.path.join(l1_path, l2, 'BUILD'), 'w') as f:
              f.write('filegroup(name="corrupted")\n')
            found = True
    self.assertTrue(found, 'failed to find cache entry to corrupt')

    _, _, stderr = self.RunBazel(['build', '@my_repo//:haha'])
    stderr = '\n'.join(stderr)
    if self.checkExternalRepositoryFiles():
      # Modification is detected and triggers a refetch with a warning.
      self.assertIn('JUST FETCHED', stderr)
      self.assertIn('WARNING: Repository \'@@+repo+my_repo\' will be fetched again since the file', stderr)
    else:
      # Modification is not detected while the server is running.
      self.assertNotIn('JUST FETCHED', stderr)
      self.assertNotIn('WARNING', stderr)

      # Modification is picked up after server restart.
      self.RunBazel(['shutdown'])
      self.RunBazel(['build', '@my_repo//:corrupted'])


if __name__ == '__main__':
  absltest.main()
