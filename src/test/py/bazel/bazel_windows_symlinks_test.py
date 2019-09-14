# Copyright 2017 The Bazel Authors. All rights reserved.
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
import unittest
from src.test.py.bazel import test_base


class BazelWindowsTest(test_base.TestBase):

    def createProjectFiles(self):
        self.CreateWorkspaceWithDefaultRepos('WORKSPACE')
        self.ScratchFile(
            'foo/BUILD', ['genrule(name="x", srcs=[":sample"], outs=["link"], cmd="python -c \\"import os; os.symlink(os.path.join(os.getcwd(), \'$<\'),\'$@\')\\"",)'])
        self.ScratchFile('foo/sample', [
            'sample',
        ])

    def testWindowsSymlinkedOutput(self):
        self.createProjectFiles()

        exit_code, _, stderr = self.RunBazel([
            '--batch', 'build',
            '//foo:x',
        ])
        self.AssertExitCode(exit_code, 0, stderr)


if __name__ == '__main__':
    unittest.main()
