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

import os
from absl.testing import absltest

from src.test.py.bazel import test_base


class BazelWorkspaceTest(test_base.TestBase):

  def setUp(self):
    test_base.TestBase.setUp(self)
    self.DisableBzlmod()

  def Read(self, src_path):
    with open(src_path, 'r') as s:
      return s.readlines()

  def testWorkspaceDotBazelFileInMainRepo(self):
    # Make sure no existing MODULE.bazel file.
    os.remove("MODULE.bazel")
    rules_python = [
                       "load('@bazel_tools//tools/build_defs/repo:http.bzl', 'http_archive')"] + self.Read(
        self.Rlocation(
            "io_bazel/src/test/py/bazel/default_repos_stanza.txt")) + [
                   ] + [
                       "load('@rules_python//python:repositories.bzl', 'py_repositories')",
                       'py_repositories()'
                   ]
    workspace_dot_bazel = self.ScratchFile("WORKSPACE.bazel", rules_python)
    self.ScratchFile(
        "BUILD",
        [
            "load('@rules_python//python:py_binary.bzl', 'py_binary')",
            "py_binary(",
            "  name = 'bin',",
            "  srcs = ['bin.py'],",
            ")",
        ],
    )
    self.ScratchFile("bin.py")
    self.RunBazel(["build", "--incompatible_autoload_externally=", "//:bin"])

    # If WORKSPACE.bazel is deleted and no WORKSPACE exists,
    # the build should fail.
    os.remove(workspace_dot_bazel)
    exit_code, _, stderr = self.RunBazel(
        ["build", "//:bin"], allow_failure=True
    )
    # this test is supposed to fail
    self.AssertExitCode(exit_code, 2, stderr)

  def testWorkspaceDotBazelFileWithExternalRepo(self):
    self.ScratchDir("A")
    rules_python = [
                       "load('@bazel_tools//tools/build_defs/repo:http.bzl', 'http_archive')"] + self.Read(
        self.Rlocation(
            "io_bazel/src/test/py/bazel/default_repos_stanza.txt")) + [
                   ] + [
                       "load('@rules_python//python:repositories.bzl', 'py_repositories')",
                       'py_repositories()'
                   ]
    self.ScratchFile("A/WORKSPACE.bazel", rules_python)
    self.ScratchFile("A/BUILD", [
        "load('@rules_python//python:py_library.bzl', 'py_library')",
        "py_library(",
        "  name = 'lib',",
        "  srcs = ['lib.py'],",
        "  visibility = ['//visibility:public'],",
        ")",
    ])
    self.ScratchFile("A/lib.py")
    work_dir = self.ScratchDir("B")
    # Test WORKSPACE.bazel takes priority over WORKSPACE
    self.ScratchFile("B/WORKSPACE", rules_python)
    workspace_dot_bazel = self.ScratchFile(
        "B/WORKSPACE.bazel",
        rules_python + [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            "local_repository(name = 'A', path='../A')",
        ],
    )
    self.ScratchFile("B/bin.py")
    self.ScratchFile("B/BUILD", [
        "load('@rules_python//python:py_binary.bzl', 'py_binary')",
        "py_binary(",
        "  name = 'bin',",
        "  srcs = ['bin.py'],",
        "  deps = ['@A//:lib'],",
        ")",
    ])
    self.RunBazel(args=["build", "--incompatible_autoload_externally=", ":bin"],
                  cwd=work_dir)

    # Test WORKSPACE takes effect after deleting WORKSPACE.bazel
    os.remove(workspace_dot_bazel)
    exit_code, _, stderr = self.RunBazel(
        args=["build", ":bin"], cwd=work_dir, allow_failure=True
    )
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn("no such package '@@A//'", "".join(stderr))

    # Test a WORKSPACE.bazel directory won't confuse Bazel
    self.ScratchFile(
        "B/WORKSPACE",
        rules_python + [
            (
                'load("@bazel_tools//tools/build_defs/repo:local.bzl",'
                ' "local_repository")'
            ),
            "local_repository(name = 'A', path='../A')",
        ],
    )
    self.ScratchDir("B/WORKSPACE.bazel")
    self.RunBazel(args=["build", "--incompatible_autoload_externally=", ":bin"],
                  cwd=work_dir)


if __name__ == "__main__":
  absltest.main()
