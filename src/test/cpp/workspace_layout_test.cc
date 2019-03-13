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
#include "src/main/cpp/util/path.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

class WorkspaceLayoutTest : public ::testing::Test {
 protected:
  WorkspaceLayoutTest() :
      build_root_(blaze_util::JoinPath(
          blaze::GetPathEnv("TEST_TMPDIR"), "build_root")),
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

}  // namespace blaze
