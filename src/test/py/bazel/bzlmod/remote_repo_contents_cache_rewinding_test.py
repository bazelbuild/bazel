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

import os
from absl.testing import absltest
from src.test.py.bazel.bzlmod import remote_repo_contents_cache_test_base


class RemoteRepoContentsCacheRewindingTest(
    remote_repo_contents_cache_test_base.RemoteRepoContentsCacheTestBase
):
  """Tests recovery of repo files lost from the remote cache."""

  def WorkerArgs(self):
    # The remote repo contents cache has to cope with caches that serve action
    # results without verifying that the blobs they reference are still
    # present, which is what makes a repo's cached Tree outlive its file
    # contents in the first place.
    return ['--noaction_cache_integrity_check']

  def BazelrcLines(self):
    # Files lost from the remote repo contents cache are recovered by
    # rewinding their repo fetch within a single command, so the tests must
    # not be rescued by a retry of the whole command.
    return super().BazelrcLines() + [
        'common --rewind_lost_inputs',
        'common --experimental_remote_cache_eviction_retries=0',
    ]

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
    # Then, lose all remote files and build a target in the other build file.
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

    # Build the other target: its BUILD file is no longer available remotely
    # and is recovered by rewinding the repo fetch.
    _, _, stderr = self.RunBazel(['build', '@my_repo//sub:sub'])
    stderr = '\n'.join(stderr)
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

  def testLostRemoteFile_query(self):
    # Like testLostRemoteFile_build, but the lost BUILD file is read by a
    # command that doesn't build and thus has no --rewind_lost_inputs to go by.
    # Such a command never executes actions, so rewinding the repo fetch is
    # always safe for it.
    repo_dir = self._setupRepoWithSubpackage()

    # First fetch: not cached
    _, _, stderr = self.RunBazel(['build', '@my_repo//:root'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging: cached
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:root'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))

    # Lose all remote files.
    self.ClearRemoteCache()

    # Query the other package: its BUILD file is no longer available remotely
    # and is recovered by rewinding the repo fetch.
    _, stdout, stderr = self.RunBazel(['query', '@my_repo//sub:all'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertIn('@my_repo//sub:sub', '\n'.join(stdout))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'sub/BUILD')))

  def testLostRemoteFile_bazelignore_prefetched(self):
    self.ScratchFile('MODULE.bazel', [
        'repo = use_repo_rule("//:repo.bzl", "repo")',
        'repo(name = "my_repo")',
    ])
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile('repo.bzl', [
        'def _repo_impl(rctx):',
        '  rctx.file("BUILD", "filegroup(name=\'root\')")',
        '  rctx.file(".bazelignore", "ignored\\n")',
        '  rctx.file("ignored/BUILD", "this is not a valid BUILD file")',
        '  print("JUST FETCHED")',
        '  return rctx.repo_metadata(reproducible=True)',
        'repo = repository_rule(_repo_impl)',
    ])
    repo_dir = self.RepoDir('my_repo')
    self.RunBazel(['build', '@my_repo//...'])
    self.RunBazel(['clean', '--expunge'])
    # Preserve the cached tree and action result, but lose its .bazelignore.
    self.DeleteCasEntry(b'ignored\n')
    _, _, stderr = self.RunBazel(['build', '@my_repo//...'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, '.bazelignore')))

    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//...'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    # Even on a cache hit, later package loads need no lazy read of this file.
    self.assertTrue(os.path.exists(os.path.join(repo_dir, '.bazelignore')))

  def testLostRemoteFile_actionInput_rewound(self):
    self._testLostActionInput(symlink=False)

  def testLostRemoteFile_actionInput_belowDirectorySymlink(self):
    self._testLostActionInput(symlink=True)

  def testLostRemoteFile_actionInput_retriedWithoutRewinding(self):
    self._testLostActionInput(symlink=False, rewind=False)

  def testLostRemoteFile_actionInput_behindSymlinkAction(self):
    # The genrule consumes the output of a symlink action pointing at the
    # repo file rather than the repo file itself. The lost input reported is
    # that output, whose contents are those of the repo file.
    self._testLostActionInput(symlink=False, through_symlink_action=True)

  def testLostRemoteFile_symlinkAction_belowDirectorySymlink(self):
    self._testLostActionInput(symlink=True, through_symlink_action=True)

  def testLostRemoteFile_symlinkAction_retriedWithoutRewinding(self):
    self._testLostActionInput(
        symlink=False, rewind=False, through_symlink_action=True
    )

  def testLostRemoteFile_topLevelSymlink_belowDirectorySymlink(self):
    self._testLostActionInput(
        symlink=True, through_symlink_action=True, top_level=True
    )

  def testLostRemoteFile_topLevelSymlink_retriedWithoutRewinding(self):
    self._testLostActionInput(
        symlink=False, rewind=False, through_symlink_action=True, top_level=True
    )

  def testLostRemoteFile_topLevelSymlink_completionHandler_rewound(self):
    self._testLostActionInput(
        symlink=True,
        through_symlink_action=True,
        top_level=True,
        finalize_actions=False,
    )

  def testLostRemoteFile_topLevelSymlink_completionHandler_retried(self):
    self._testLostActionInput(
        symlink=False,
        rewind=False,
        through_symlink_action=True,
        top_level=True,
        finalize_actions=False,
    )

  def _testLostActionInput(
      self,
      symlink,
      rewind=True,
      through_symlink_action=False,
      top_level=False,
      finalize_actions=True,
  ):
    data_path = 'link/data.txt' if symlink else 'data.txt'
    real_path = 'real/data.txt' if symlink else data_path
    target = '//main:link' if top_level else '//main:use_data'
    output_path = 'bazel-bin/main/link.txt' if top_level else 'bazel-bin/main/out.txt'
    if not finalize_actions:
      # Let the completion handler perform the download, so it observes the
      # eviction rather than the symlink action's finalization.
      self.ScratchFile('.bazelrc', ['build --nooutput_tree_tracking'], mode='a')
    # Create a repo with a data file consumed by a genrule, cache the repo
    # remotely, then build the genrule after the remote cache lost all files.
    # With --rewind_lost_inputs, the lost action input is recovered by
    # rewinding, which refetches the repo within the same command. Without it,
    # the command fails and is retried as a whole, which refetches the repo.
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
            f'  rctx.file("BUILD", "exports_files([\'{data_path}\'])")',
            f'  rctx.file("{real_path}", "hello")',
            '  rctx.symlink("real", "link")' if symlink else '',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    if through_symlink_action:
      self.ScratchFile(
          'main/symlink.bzl',
          [
              'def _symlink_impl(ctx):',
              '  out = ctx.actions.declare_file(ctx.label.name + ".txt")',
              '  ctx.actions.symlink(output = out, target_file = ctx.file.src)',
              '  return [DefaultInfo(files = depset([out]))]',
              'symlink = rule(',
              '  implementation = _symlink_impl,',
              '  attrs = {"src": attr.label(allow_single_file = True)},',
              ')',
          ],
      )
      build_lines = [
          'load("//main:symlink.bzl", "symlink")',
          f'symlink(name = "link", src = "@my_repo//:{data_path}")',
      ]
      src = ':link'
    else:
      build_lines = []
      src = f'@my_repo//:{data_path}'
    self.ScratchFile(
        'main/BUILD.bazel',
        build_lines
        + [
            'genrule(',
            '  name = "use_data",',
            f'  srcs = ["{src}"],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $(SRCS) > $@",',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached. Analyze (but do not execute) the target so
    # that all loading and analysis state is in Skyframe for the builds below.
    _, _, stderr = self.RunBazel(['build', '--nobuild', target])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, data_path)))

    # After expunging: cached, with the contents of data.txt staying remote.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '--nobuild', target])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, data_path)))

    # Lose all remote files.
    self.ClearRemoteCache()

    # The genrule's input, or the top-level symlink's target, is lost remotely.
    if rewind:
      _, _, stderr = self.RunBazel(
          ['build', '--rewind_lost_inputs', target]
      )
      stderr = '\n'.join(stderr)
      self.assertNotIn('retrying the build', stderr)
    else:
      _, _, stderr = self.RunBazel([
          'build',
          '--norewind_lost_inputs',
          '--experimental_remote_cache_eviction_retries=5',
          target,
      ])
      stderr = '\n'.join(stderr)
      self.assertEqual(
          1,
          stderr.count(
              'Found transient remote cache error, retrying the build...'
          ),
      )
    self.assertIn('JUST FETCHED', stderr)
    # The refetch materializes the repo on disk.
    self.assertTrue(os.path.exists(os.path.join(repo_dir, data_path)))
    with open(self.Path(output_path)) as f:
      self.assertEqual(f.read().strip(), 'hello')

    # After expunging again: cached, with the repo contents having been
    # uploaded again by the refetch.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', target])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    if not top_level:
      # The genrule's cached output needs no source download; a top-level
      # symlink still needs its source target to be downloaded.
      self.assertFalse(os.path.exists(os.path.join(repo_dir, data_path)))
    with open(self.Path(output_path)) as f:
      self.assertEqual(f.read().strip(), 'hello')

  def testLostRemoteFile_actionInputs_multipleFilesFromSameRepo(self):
    # Two files of the same cached repo are lost from the remote cache and
    # consumed by a single action. Since the repo rule that produced them can
    # only be run as a whole, a single refetch has to recover both.
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
                '  rctx.file("BUILD",'
                ' "exports_files([\'data_1.txt\', \'data_2.txt\'])")'
            ),
            '  rctx.file("data_1.txt", "unique-contents-1\\n")',
            '  rctx.file("data_2.txt", "unique-contents-2\\n")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "use_both",',
            '  srcs = [',
            '    "@my_repo//:data_1.txt",',
            '    "@my_repo//:data_2.txt",',
            '  ],',
            '  outs = ["out.txt"],',
            '  cmd = "cat $(SRCS) > $@",',
            ')',
        ],
    )

    repo_dir = self.RepoDir('my_repo')

    # First fetch: not cached. Analyze (but do not execute) the genrule so
    # that all loading and analysis state is in Skyframe for the builds below.
    _, _, stderr = self.RunBazel(['build', '--nobuild', '//main:use_both'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # After expunging: cached, with the contents of both data files staying
    # remote.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '--nobuild', '//main:use_both'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data_1.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data_2.txt')))

    # Lose the blobs of both data files while keeping the repo's action result
    # and Tree, so that the loss is only discovered when the action's inputs
    # are materialized.
    self.DeleteCasEntry(b'unique-contents-1\n')
    self.DeleteCasEntry(b'unique-contents-2\n')

    _, _, stderr = self.RunBazel(
        ['build', '--rewind_lost_inputs', '//main:use_both']
    )
    # A single refetch recovered both lost files.
    self.assertEqual('\n'.join(stderr).count('JUST FETCHED'), 1)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data_1.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data_2.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'unique-contents-1\nunique-contents-2\n')

    # The refetch uploaded the repo contents anew, which healed the cache
    # entry for both lost blobs, not just for the one that surfaced first.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:use_both'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data_1.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data_2.txt')))
    with open(self.Path('bazel-bin/main/out.txt')) as f:
      self.assertEqual(f.read(), 'unique-contents-1\nunique-contents-2\n')

  def SetUpRemoteOnlyDataRepo(self):
    """Caches @my_repo and restores it with data.txt remaining remote-only."""
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD.bazel",'
                ' "exports_files([\'data.txt\'])\\n'
                'filegroup(name=\'metadata_only\')")'
            ),
            '  rctx.file("data.txt", "unique-data-file-contents")',
            '  print("JUST FETCHED")',
            '  return rctx.repo_metadata(reproducible=True)',
            'repo = repository_rule(_repo_impl)',
        ],
    )

    repo_dir = self.RepoDir('my_repo')
    _, _, stderr = self.RunBazel(['build', '@my_repo//:metadata_only'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:metadata_only'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    return repo_dir

  def ScratchOtherRepoReadingData(self):
    """Writes the rule for @other, which copies @my_repo//:data.txt."""
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            (
                '  rctx.file("BUILD.bazel",'
                ' "exports_files([\'copy.txt\'])")'
            ),
            '  rctx.file("copy.txt", rctx.read(rctx.path(rctx.attr.data)))',
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"data": attr.label()})'
            ),
        ],
    )

  def testLostRemoteFile_fullMaterialization(self):
    # Regression test for https://github.com/bazelbuild/bazel/issues/30218:
    # a cached repo whose Tree references a CAS blob that is no longer
    # available must be discarded and refetched when another repo rule
    # triggers its full materialization via rctx.path()/rctx.read().
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", data = "@my_repo//:data.txt")',
        ],
    )
    self.ScratchFile(
        'other_repo.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.file("BUILD.bazel", "filegroup(name=\'copy\')")',
            '  rctx.file("copy.txt", rctx.read(rctx.path(rctx.attr.data)))',
            '  return rctx.repo_metadata()',
            (
                'other_repo = repository_rule(_other_repo_impl,'
                ' attrs={"data": attr.label()})'
            ),
        ],
    )

    repo_dir = self.SetUpRemoteOnlyDataRepo()

    # Delete the CAS blob for data.txt while keeping the repo's action result
    # and Tree. Building @other, whose repo rule reads data.txt through
    # rctx.path()/rctx.read(), forces the full materialization of @my_repo,
    # which discovers the lost file. The unusable cache entry must be
    # discarded and the repo rule run again.
    self.DeleteCasEntry(b'unique-data-file-contents')
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

    # The refetch has healed the cache entry: after expunging, the repo is
    # restored from the cache and can be fully materialized again.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

  def testLostRemoteFile_moduleExtensionMaterialization(self):
    # Like testLostRemoteFile_fullMaterialization, but with the full
    # materialization triggered by a module extension via module_ctx.path(),
    # whose failures are reported through module extension evaluation rather
    # than package lookup.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'ext = use_extension("//:extension.bzl", "ext")',
            'use_repo(ext, "other")',
        ],
    )
    self.ScratchFile(
        'extension.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.file("BUILD.bazel", "filegroup(name=\'copy\')")',
            'other_repo = repository_rule(_other_repo_impl)',
            'def _ext_impl(module_ctx):',
            '  module_ctx.path(Label("@my_repo//:data.txt"))',
            '  other_repo(name = "other")',
            'ext = module_extension(_ext_impl)',
        ],
    )

    repo_dir = self.SetUpRemoteOnlyDataRepo()

    # Delete the CAS blob for data.txt while keeping the repo's action result
    # and Tree. Building @other, whose module extension accesses data.txt
    # through module_ctx.path(), forces the full materialization of @my_repo,
    # which discovers the lost file. The unusable cache entry must be
    # discarded and the repo rule run again.
    self.DeleteCasEntry(b'unique-data-file-contents')
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

    # The refetch has healed the cache entry: after expunging, the repo is
    # restored from the cache and can be fully materialized again. The
    # lockfile has to be removed as it would otherwise short-circuit the
    # extension evaluation and thus the materialization.
    self.RunBazel(['clean', '--expunge'])
    os.remove(self.Path('MODULE.bazel.lock'))
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

  def testLostRemoteFile_multipleReposRecoverInOneBuild(self):
    # Two cached repos independently reference a lost CAS blob, but the second
    # one is only reached after the first has recovered: a module extension
    # materializes them one after the other and aborts at the first failure.
    # Both repos must still recover within a single build.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "repo_a", marker = "a")',
            'repo(name = "repo_b", marker = "b")',
            'ext = use_extension("//:extension.bzl", "ext")',
            'use_repo(ext, "other")',
        ],
    )
    self.ScratchFile('BUILD.bazel')
    self.ScratchFile(
        'repo.bzl',
        [
            'def _repo_impl(rctx):',
            (
                '  rctx.file("BUILD.bazel",'
                ' "exports_files([\'data.txt\'])\\n'
                'filegroup(name=\'metadata_only\')")'
            ),
            (
                '  rctx.file("data.txt",'
                ' "unique-data-file-contents-" + rctx.attr.marker)'
            ),
            '  print("JUST FETCHED " + rctx.attr.marker)',
            '  return rctx.repo_metadata(reproducible=True)',
            (
                'repo = repository_rule(_repo_impl,'
                ' attrs={"marker": attr.string(mandatory=True)})'
            ),
        ],
    )
    # Extension evaluation is sequential and aborts at the first failure, so
    # @repo_b is only materialized once @repo_a is healthy again.
    self.ScratchFile(
        'extension.bzl',
        [
            'def _other_repo_impl(rctx):',
            '  rctx.file("BUILD.bazel", "filegroup(name=\'copy\')")',
            'other_repo = repository_rule(_other_repo_impl)',
            'def _ext_impl(module_ctx):',
            '  module_ctx.path(Label("@repo_a//:data.txt"))',
            '  module_ctx.path(Label("@repo_b//:data.txt"))',
            '  other_repo(name = "other")',
            'ext = module_extension(_ext_impl)',
        ],
    )

    repo_a_dir = self.RepoDir('repo_a')
    repo_b_dir = self.RepoDir('repo_b')
    metadata_targets = [
        '@repo_a//:metadata_only',
        '@repo_b//:metadata_only',
    ]

    # Populate the remote repo contents cache, then restore only the repo
    # metadata into the in-memory overlay. data.txt remains remote-only.
    _, _, stderr = self.RunBazel(['build'] + metadata_targets)
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED a', stderr)
    self.assertIn('JUST FETCHED b', stderr)
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build'] + metadata_targets)
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(os.path.exists(os.path.join(repo_a_dir, 'data.txt')))
    self.assertFalse(os.path.exists(os.path.join(repo_b_dir, 'data.txt')))

    self.DeleteCasEntry(b'unique-data-file-contents-a')
    self.DeleteCasEntry(b'unique-data-file-contents-b')
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED a', stderr)
    self.assertIn('JUST FETCHED b', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_a_dir, 'data.txt')))
    self.assertTrue(os.path.exists(os.path.join(repo_b_dir, 'data.txt')))

    # Both cache entries have been healed by the refetch.
    self.RunBazel(['clean', '--expunge'])
    os.remove(self.Path('MODULE.bazel.lock'))
    _, _, stderr = self.RunBazel(['build', '@other//:copy'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)

  def testLostRemoteFile_sourceDirectoryMaterialization(self):
    # Like testLostRemoteFile_fullMaterialization, but with only the subtree
    # below a source directory input materialized for a local action. The only
    # lost file lies within that subtree; all other files, including the BUILD
    # file read during loading, remain available in the remote cache.
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
                "filegroup(name='metadata_only')\")"
            ),
            (
                '  rctx.file("sysroot/include/data.txt",'
                ' "unique-source-dir-contents")'
            ),
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

    # Populate the remote repo contents cache.
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    self.assertIn('JUST FETCHED', '\n'.join(stderr))

    # Restore only the repo metadata into the in-memory overlay. All files,
    # including those below the source directory, remain remote-only.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '@my_repo//:metadata_only'])
    self.assertNotIn('JUST FETCHED', '\n'.join(stderr))
    self.assertFalse(
        os.path.exists(os.path.join(repo_dir, 'sysroot/include/data.txt'))
    )

    # Delete the CAS blob for data.txt while keeping the repo's action result,
    # Tree, and all other blobs. The local genrule action triggers the
    # materialization of only the sysroot subtree, which discovers the lost
    # file. The unusable cache entry must be discarded and the repo rule run
    # again.
    self.DeleteCasEntry(b'unique-source-dir-contents')
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(
        os.path.exists(os.path.join(repo_dir, 'sysroot/include/data.txt'))
    )
    with open(out) as f:
      self.assertEqual(f.read(), 'unique-source-dir-contents')

    # The refetch has healed the cache entry: after expunging, the repo is
    # restored from the cache and the subtree can be materialized again.
    self.RunBazel(['clean', '--expunge'])
    _, _, stderr = self.RunBazel(['build', '//main:read_source_directory'])
    stderr = '\n'.join(stderr)
    self.assertNotIn('JUST FETCHED', stderr)
    with open(out) as f:
      self.assertEqual(f.read(), 'unique-source-dir-contents')

  def doTestLostRemoteFile_analysisMaterialization(
      self, *, keep_going, nobuild=False
  ):
    # A lost blob is discovered while fetching a repo that is only reachable
    # via a dependency edge, i.e. during the analysis of the top-level target
    # rather than during target pattern expansion. The repo fetch is rewound
    # and recovers before any error reaches the top-level target, with and
    # without --keep_going.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", data = "@my_repo//:data.txt")',
        ],
    )
    self.ScratchOtherRepoReadingData()
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "bin",',
            '  srcs = ["@other//:copy.txt"],',
            '  outs = ["bin.txt"],',
            '  cmd = "cat $< > $@",',
            ')',
        ],
    )

    repo_dir = self.SetUpRemoteOnlyDataRepo()

    # Delete the CAS blob for data.txt while keeping the repo's action result
    # and Tree. @other is fetched while analyzing //main:bin, which depends on
    # it, and its repo rule reads data.txt through rctx.path()/rctx.read().
    # The lost file must be recovered during that fetch, before the analysis
    # of //main:bin could fail.
    self.DeleteCasEntry(b'unique-data-file-contents')
    args = ['build']
    if keep_going:
      args.append('--keep_going')
    if nobuild:
      args.append('--nobuild')
    args.append('//main:bin')
    _, _, stderr = self.RunBazel(args)
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    if not nobuild:
      with open(self.Path('bazel-bin/main/bin.txt')) as f:
        self.assertEqual(f.read(), 'unique-data-file-contents')

  def testLostRemoteFile_analysisMaterialization(self):
    self.doTestLostRemoteFile_analysisMaterialization(keep_going=False)

  def testLostRemoteFile_analysisMaterialization_keepGoing(self):
    self.doTestLostRemoteFile_analysisMaterialization(keep_going=True)

  # --nobuild disables Skymeld, which reports analysis errors through a
  # different code path than the merged analysis and execution phase.
  def testLostRemoteFile_analysisMaterialization_noBuild(self):
    self.doTestLostRemoteFile_analysisMaterialization(
        keep_going=False, nobuild=True
    )

  def testLostRemoteFile_analysisMaterialization_keepGoing_noBuild(self):
    self.doTestLostRemoteFile_analysisMaterialization(
        keep_going=True, nobuild=True
    )

  def doTestLostRemoteFile_aspectMaterialization(
      self, *, keep_going, nobuild=False
  ):
    # Like doTestLostRemoteFile_analysisMaterialization, but the repo with the
    # lost blob is only reachable through an implicit attribute of a top-level
    # aspect, so the fetch is triggered by the analysis of the aspect rather
    # than of a configured target and must recover just the same.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", data = "@my_repo//:data.txt")',
        ],
    )
    self.ScratchOtherRepoReadingData()
    # Only the aspect depends on @other, through its implicit attribute, so the
    # base target analyzes without triggering the affected fetch.
    self.ScratchFile(
        'main/aspect.bzl',
        [
            'def _my_aspect_impl(target, ctx):',
            '  return []',
            'my_aspect = aspect(',
            '  implementation = _my_aspect_impl,',
            '  attrs = {',
            (
                '    "_tool": attr.label(default ='
                ' Label("@other//:copy.txt"), allow_single_file = True),'
            ),
            '  },',
            ')',
        ],
    )
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "plain",',
            '  outs = ["plain.txt"],',
            '  cmd = "printf plain > $@",',
            ')',
        ],
    )

    repo_dir = self.SetUpRemoteOnlyDataRepo()

    self.DeleteCasEntry(b'unique-data-file-contents')
    args = ['build', '--aspects=//main:aspect.bzl%my_aspect']
    if keep_going:
      args.append('--keep_going')
    if nobuild:
      args.append('--nobuild')
    args.append('//main:plain')
    _, _, stderr = self.RunBazel(args)
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))

  def testLostRemoteFile_analysisMaterialization_keepGoing_withHealthyTarget(self):
    # A --keep_going build whose other target analyzes and executes normally,
    # so that the recovering repo fetch runs alongside action execution.
    self.ScratchFile(
        'MODULE.bazel',
        [
            'repo = use_repo_rule("//:repo.bzl", "repo")',
            'repo(name = "my_repo")',
            'other_repo = use_repo_rule("//:other_repo.bzl", "other_repo")',
            'other_repo(name = "other", data = "@my_repo//:data.txt")',
        ],
    )
    self.ScratchOtherRepoReadingData()
    self.ScratchFile(
        'main/BUILD.bazel',
        [
            'genrule(',
            '  name = "bin",',
            '  srcs = ["@other//:copy.txt"],',
            '  outs = ["bin.txt"],',
            '  cmd = "cat $< > $@",',
            ')',
            'genrule(',
            '  name = "healthy",',
            '  outs = ["healthy.txt"],',
            '  cmd = "printf healthy > $@",',
            ')',
        ],
    )

    repo_dir = self.SetUpRemoteOnlyDataRepo()

    self.DeleteCasEntry(b'unique-data-file-contents')
    _, _, stderr = self.RunBazel(
        ['build', '--keep_going', '//main:bin', '//main:healthy']
    )
    stderr = '\n'.join(stderr)
    self.assertIn('JUST FETCHED', stderr)
    self.assertTrue(os.path.exists(os.path.join(repo_dir, 'data.txt')))
    with open(self.Path('bazel-bin/main/bin.txt')) as f:
      self.assertEqual(f.read(), 'unique-data-file-contents')
    with open(self.Path('bazel-bin/main/healthy.txt')) as f:
      self.assertEqual(f.read(), 'healthy')

  def testLostRemoteFile_aspectMaterialization(self):
    self.doTestLostRemoteFile_aspectMaterialization(keep_going=False)

  def testLostRemoteFile_aspectMaterialization_keepGoing(self):
    self.doTestLostRemoteFile_aspectMaterialization(keep_going=True)

  # --nobuild disables Skymeld, which reports analysis errors through a
  # different code path than the merged analysis and execution phase.
  def testLostRemoteFile_aspectMaterialization_noBuild(self):
    self.doTestLostRemoteFile_aspectMaterialization(
        keep_going=False, nobuild=True
    )

  def testLostRemoteFile_aspectMaterialization_keepGoing_noBuild(self):
    self.doTestLostRemoteFile_aspectMaterialization(
        keep_going=True, nobuild=True
    )


if __name__ == '__main__':
  absltest.main()
