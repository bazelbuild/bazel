// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/main/cpp/workspace_layout.h"

#include <fcntl.h>

#include <memory>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

class WorkspaceLayoutTest : public ::testing::Test {
 protected:
  WorkspaceLayoutTest() :
      build_root_(blaze_util::JoinPath(
          blaze::GetEnv("TEST_TMPDIR"), "build_root")),
      workspace_layout_(new WorkspaceLayout()) {}

  void SetUp() override {
    ASSERT_TRUE(blaze_util::MakeDirectories(build_root_, 0755));
    ASSERT_TRUE(blaze_util::WriteFile(
        "", blaze_util::JoinPath(build_root_, "WORKSPACE"), 0755));
  }

  void TearDown() override {
    // TODO(bazel-team): The code below deletes all the files in the workspace
    // but it intentionally skips directories. As a consequence, there may be
    // empty directories from test to test. Remove this once
    // blaze_util::DeleteDirectories(path) exists.
    std::vector<std::string> files_in_workspace;
    blaze_util::GetAllFilesUnder(build_root_, &files_in_workspace);
    for (const std::string& file : files_in_workspace) {
      blaze_util::UnlinkPath(file);
    }
  }

  const std::string build_root_;
  const std::unique_ptr<WorkspaceLayout> workspace_layout_;
};

TEST_F(WorkspaceLayoutTest, GetWorkspace) {
  // "" is returned when there's no workspace path.
  std::string cwd = "foo/bar";
  ASSERT_EQ("", workspace_layout_->GetWorkspace(cwd));
  ASSERT_FALSE(workspace_layout_->InWorkspace(cwd));

  cwd = build_root_;
  ASSERT_EQ(build_root_, workspace_layout_->GetWorkspace(cwd));
  ASSERT_TRUE(workspace_layout_->InWorkspace(build_root_));

  cwd = blaze_util::JoinPath(build_root_, cwd);
  ASSERT_EQ(build_root_, workspace_layout_->GetWorkspace(cwd));
}

TEST_F(WorkspaceLayoutTest, FindCandidateBlazercPaths) {
  const std::string binary_dir = blaze_util::JoinPath(build_root_, "bazeldir");
  const std::string tools_dir = blaze_util::JoinPath(build_root_, "tools");
  const std::string workspace_rc_path =
      blaze_util::JoinPath(build_root_, "tools/bazel.rc");
  const std::string binary_rc_path =
      blaze_util::JoinPath(binary_dir, "bazel.bazelrc");
  ASSERT_TRUE(blaze_util::MakeDirectories(binary_dir, 0755));
  ASSERT_TRUE(blaze_util::MakeDirectories(tools_dir, 0755));
  ASSERT_TRUE(blaze_util::WriteFile("", workspace_rc_path, 0755));
  ASSERT_TRUE(blaze_util::WriteFile("", binary_rc_path, 0755));

  std::vector<std::string> expected = {workspace_rc_path, binary_rc_path};
  std::vector<std::string> actual =
      workspace_layout_->FindCandidateBlazercPaths(
          build_root_, build_root_, "bazeldir/bazel", {});
  // The third entry is the system wide blazerc path, /etc/bazel.bazelrc, which
  // we do not mock within this test because it is not within the sandbox. It
  // may or may not exist on the system running the test, so we do not check for
  // it.
  // TODO(https://github.com/bazelbuild/bazel/issues/4502): Make the system-wide
  // master bazelrc location configurable and add test coverage for it.
  std::vector<std::string> actual_first_two_entries(actual.begin(),
                                                    actual.begin() + 2);
  ASSERT_EQ(expected, actual_first_two_entries);
}

}  // namespace blaze
