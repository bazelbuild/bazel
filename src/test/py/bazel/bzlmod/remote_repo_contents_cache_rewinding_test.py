# Copyright 2026 The Bazel Authors. All rights reserved.
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

import hashlib
import json
import os
import re
from absl.testing import absltest
from src.test.py.bazel.bzlmod import remote_repo_contents_cache_test_base


class RemoteRepoContentsCacheRewindingTest(
    remote_repo_contents_cache_test_base.RemoteRepoContentsCacheTestBase
):
  """Tests recovery of repo files lost from the remote cache."""

  def _setupRepoWithSubpackage(self):
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

    return self.RepoDir('my_repo')

  def testLostRemoteFile_build(self):
    # Create a repo with two BUILD files (one in a subpackage), build a target
    # from one to cause it to be cached, then build that target again after
    # expunging to verify it is cached.
    # Then, restart the worker and build a target in the other build file.
    repo_dir = self._setupRepoWithSubpackage()

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
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

    # Lose all remote files.
    self.ClearRemoteCache()

    # Build the other target: fails due to the lost input
    _, _, stderr = self.RunBazel(['build', '@my_repo//sub:sub'])
    # First restart recovers @my_repo, the next one recovers @platforms.
    self.assertEqual(
        2,
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
    _, _, stderr = self.RunBazel(['build', '@my_repo//sub:sub'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'BUILD')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'root.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/sub.txt')))

  def testLostRemoteFile_build_evaluatorReplaced(self):
    # The first command that tracks incremental state after one that doesn't
    # replaces the Skyframe evaluator while it is running. The fetch of a repo
    # with lost files has to be invalidated in the new evaluator for the next
    # command to fetch the repo again. Otherwise, that only happens if the
    # files of external repos are checked for changes.
    self.ScratchFile(
        '.bazelrc',
        ['common --noexperimental_check_external_repository_files'],
        mode='a',
    )
    sub_build = "filegroup(name='sub', visibility=['//visibility:public'])"
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'deleter = use_repo_rule("//:deleter.bzl", "deleter")',
            'deleter(name = "deleter")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            '  rctx.file("BUILD", "filegroup(name=\'root\')")',
            '  rctx.file("defs.bzl", "MARKER = 1")',
            '  rctx.file("sub/BUILD", "%s")' % sub_build,
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    # A repo that is only fetched by the command that replaces the evaluator.
    # Loading the file that defines its repo rule retrieves @my_repo from the
    # cache, fetching it makes the remote cache lose a file of @my_repo that
    # is only read afterwards, when the target it refers to is loaded.
    digest = hashlib.sha256(sub_build.encode()).hexdigest()
    blob_path = os.path.join(self._cas_path, 'cas', digest[:2], digest)
    self.ScratchFile(
        'deleter.bzl',
        [
            'load("@my_repo//:defs.bzl", "MARKER")',
            'def _deleter_impl(rctx):',
            '  if rctx.os.name.startswith("windows"):',
            '    cmd = ["cmd.exe", "/c", "del", %s]'
            % json.dumps(blob_path.replace('/', '\\')),
            '  else:',
            '    cmd = ["rm", %s]' % json.dumps(blob_path),
            '  result = rctx.execute(cmd)',
            '  if result.return_code != 0:',
            '    fail(result.stderr)',
            '  rctx.file(',
            '    "BUILD",',
            '    "alias(name=\'sub\', actual=\'%s\')" % Label("@my_repo//sub"),',
            '  )',
            'deleter = repository_rule(_deleter_impl)',
        ],
    )
    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:root'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging: cached. This command doesn't track incremental state.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(
        ['build', '--notrack_incremental_state', '@my_repo//:root']
    )
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(blob_path))

    # Build the other target: fails due to the lost file
    exit_code, _, stderr = self.RunBazel(
        ['build', '@deleter//:sub'], allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)
    self.assertIn(
        'sub/BUILD with digest %s/%d is no longer available in the remote cache'
        % (digest, len(sub_build)),
        stderr,
    )

    # The next build fetches the repo again.
    _, _, stderr = self.RunBazel(['build', '@deleter//:sub'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))

  def testLostRemoteFile_remoteExecutionUpload(self):
    # Regression test for a crash when a file in a remotely cached repo has
    # been evicted after the analysis phase and this is only noticed while
    # uploading the inputs of a remotely executed action.
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
    args = [
        'build',
        '//main:use_data',
        '--spawn_strategy=remote',
        '--remote_executor=grpc://localhost:' + str(self._worker_port),
        # Rewinding of lost repo files isn't supported yet.
        '--norewind_lost_inputs',
        '--experimental_remote_cache_eviction_retries=1',
    ]
    self.RunBazel(args + ['--nobuild'])
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(args + ['--nobuild'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data.txt')))

    # Analysis is warm, so the eviction is only noticed while uploading the
    # inputs of the action.
    self.DeleteCasEntry(b'hello')
    exit_code, _, stderr = self.RunBazel(args, allow_failure=True)
    self.AssertExitCode(exit_code, 39, stderr)
    stderr = '\n'.join(stderr)
    self.assertIn(
        'Found transient remote cache error, retrying the build...', stderr
    )
    self.assertRegex(
        stderr, r'Lost inputs no longer available remotely: data.txt \(.*/5\)'
    )


if __name__ == '__main__':
  absltest.main()
