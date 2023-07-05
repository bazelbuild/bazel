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

import unittest
from src.test.py.bazel import test_base


class BazelWindowsSymlinksTest(test_base.TestBase):

  def createProjectFiles(self):
    self.CreateWorkspaceWithDefaultRepos('WORKSPACE')
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


if __name__ == '__main__':
  unittest.main()
