# pylint: disable=g-bad-file-header
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

from src.test.py.bazel import test_base


class BazelWorkspaceTest(test_base.TestBase):

  def testWorkspaceDotBazelFileInMainRepo(self):
    self.ScratchFile("WORKSPACE.bazel")
    self.ScratchFile("BUILD", [
      "py_binary(",
      "  name = 'bin',",
      "  srcs = ['bin.py'],",
      ")",
    ])
    self.ScratchFile("bin.py")
    exit_code, _, stderr = self.RunBazel(['build', '//:bin'])
    self.AssertExitCode(exit_code, 0, stderr)

  def testWorkspaceDotBazelFileInExternalRepo(self):
    self.ScratchDir("A")
    self.ScratchFile("A/WORKSPACE.bazel")
    self.ScratchFile("A/BUILD", [
      "py_library(",
      "  name = 'lib',",
      "  srcs = ['lib.py'],",
      "  visibility = ['//visibility:public'],",
      ")",
    ])
    self.ScratchFile("A/lib.py")
    work_dir = self.ScratchDir("B")
    # Test WORKSPACE.bazel takes priority over WORKSPACE
    self.ScratchFile("B/WORKSPACE")
    self.ScratchFile("B/WORKSPACE.bazel",
                     ["local_repository(name = 'A', path='../A')"])
    self.ScratchFile("B/bin.py")
    self.ScratchFile("B/BUILD", [
      "py_binary(",
      "  name = 'bin',",
      "  srcs = ['bin.py'],",
      "  deps = ['@A//:lib'],",
      ")",
    ])
    exit_code, _, stderr = self.RunBazel(args=["build", ":bin"], cwd=work_dir)
    self.AssertExitCode(exit_code, 0, stderr)
