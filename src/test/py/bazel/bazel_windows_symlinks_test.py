# Copyright 2019 The Bazel Authors. All rights reserved.
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

from absl.testing import absltest
from src.test.py.bazel import test_base


class BazelWindowsSymlinksTest(test_base.TestBase):

  def createProjectFiles(self):
    self.ScratchFile(
        'foo/BUILD',
        [
            'genrule(',
            '    name = "x",',
            '    srcs = ["sample"],',
            '    outs = ["link"],',
            '    tools = ["sym.bat"],',
            '    cmd = "$(location sym.bat) $< $@",',
            ')',
            'genrule(',
            '    name = "y",',
            '    outs = ["dangling-link"],',
            '    tools = ["sym.bat"],',
            '    cmd = "$(location sym.bat) does-not-exist $@",',
            ')',
        ],
    )
    self.ScratchFile(
        'foo/sym.bat', [
            '@set IN=%1',
            '@set OUT=%2',
            r'@mklink %OUT:/=\% %cd%\%IN:/=\%',
        ],
        executable=True)
    self.ScratchFile('foo/sample', [
        'sample',
    ])

  def testWindowsSymlinkedOutput(self):
    self.createProjectFiles()
    self.RunBazel(['build', '//foo:x'])
    exit_code, _, stderr = self.RunBazel(
        ['build', '//foo:y'], allow_failure=True
    )
    self.AssertNotExitCode(exit_code, 0, stderr)

    if not any(['is a dangling symbolic link' in l for l in stderr]):
      self.fail('FAIL:\n | stderr:\n | %s' % '\n | '.join(stderr))

  def testRelativeDirectorySymlinkNotRerunWithDiskCache(self):
    # A relative symlink to a directory is materialized as a junction, which
    # stores an absolute target. With a disk cache, actions run against the
    # remote action filesystem, which used to record the relative target, so
    # the output check reran the symlink action on every build.
    self.ScratchFile(
        'foo/defs.bzl',
        [
            'def _impl(ctx):',
            '    d = ctx.actions.declare_directory("store/pkg")',
            '    ctx.actions.run_shell(',
            '        outputs = [d],',
            '        command = "mkdir -p %s && echo hi > %s/a.txt"',
            '            % (d.path, d.path),',
            '    )',
            '    s = ctx.actions.declare_symlink("store/link")',
            '    ctx.actions.symlink(',
            '        output = s,',
            '        target_path = "pkg",',
            '        target_type = "directory",',
            '    )',
            '    return [DefaultInfo(files = depset([d, s]))]',
            '',
            'dir_link = rule(implementation = _impl)',
        ],
    )
    self.ScratchFile(
        'foo/BUILD',
        ['load(":defs.bzl", "dir_link")', 'dir_link(name = "link")'],
    )
    disk_cache = self.ScratchDir('disk_cache')
    explain = self.Path('explain.txt')
    args = [
        'build',
        '--disk_cache=' + disk_cache,
        '--experimental_check_output_files',
        '//foo:link',
    ]
    self.RunBazel(args)
    self.RunBazel(args + ['--explain=' + explain])
    with open(explain, 'r') as f:
      rerun = [l for l in f if 'Creating symlink' in l]
    self.assertEqual(rerun, [])

  def testRelativeDirectorySymlinkSharedAcrossWorkspacesWithDiskCache(self):
    # The metadata of a relative symlink to a directory, which is materialized
    # as a junction with an absolute target, must not depend on the location of
    # the workspace, or else actions consuming it can't be shared through a
    # cache between workspaces at different paths.
    workspaces = ['ws1', 'other/ws2']
    for ws in workspaces:
      self.ScratchFile(ws + '/MODULE.bazel')
      self.CopyFile(
          self.Rlocation('io_bazel/src/test/tools/bzlmod/MODULE.bazel.lock'),
          ws + '/MODULE.bazel.lock',
      )
      self.ScratchFile(
          ws + '/foo/defs.bzl',
          [
              'def _impl(ctx):',
              '    d = ctx.actions.declare_directory("store/pkg")',
              '    ctx.actions.run_shell(',
              '        outputs = [d],',
              '        command = "mkdir -p %s && echo hi > %s/a.txt"',
              '            % (d.path, d.path),',
              '    )',
              '    s = ctx.actions.declare_symlink("store/link")',
              '    ctx.actions.symlink(',
              '        output = s,',
              '        target_path = "pkg",',
              '        target_type = "directory",',
              '    )',
              '    out = ctx.actions.declare_file("out.txt")',
              '    ctx.actions.run_shell(',
              '        inputs = [d, s],',
              '        outputs = [out],',
              '        command = "cat %s/a.txt > %s" % (s.path, out.path),',
              '    )',
              '    return [DefaultInfo(files = depset([out]))]',
              '',
              'dir_link = rule(implementation = _impl)',
          ],
      )
      self.ScratchFile(
          ws + '/foo/BUILD',
          ['load(":defs.bzl", "dir_link")', 'dir_link(name = "link")'],
      )
    disk_cache = self.ScratchDir('disk_cache')
    args = ['build', '--disk_cache=' + disk_cache, '//foo:link']
    try:
      self.RunBazel(args, cwd=self.Path(workspaces[0]))
      _, _, stderr = self.RunBazel(args, cwd=self.Path(workspaces[1]))
    finally:
      for ws in workspaces:
        self.RunBazel(['shutdown'], cwd=self.Path(ws))
    # Both the directory and the action consuming the symlink are cache hits.
    self.assertIn('2 disk cache hit', '\n'.join(stderr))
    with open(self.Path(workspaces[1] + '/bazel-bin/foo/out.txt'), 'r') as f:
      self.assertEqual(f.read().strip(), 'hi')


if __name__ == '__main__':
  absltest.main()
