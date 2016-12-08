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

#include "third_party/bazel/src/main/cpp/workspace_layout.h"

#include <fcntl.h>

#include "file/base/file.h"
#include "file/base/filesystem.h"
#include "file/base/helpers.h"
#include "file/base/path.h"
#include "strings/strcat.h"
#include "gtest/gtest.h"
#include "third_party/bazel/src/main/cpp/util/file.h"

namespace blaze {

class WorkspaceLayoutTest : public ::testing::Test {
 protected:
  WorkspaceLayoutTest() : workspace_layout_(new WorkspaceLayout()) {}

  void SetUp() {
    build_root_ = file::JoinPath(FLAGS_test_tmpdir, "build_root");
    CHECK_OK(RecursivelyCreateDir(build_root_, file::Defaults()));
    CHECK_OK(file::SetContents(
        file::JoinPath(build_root_, "WORKSPACE"), "", file::Defaults()));

    // Create fake javac so that Blaze can find the javabase
    string javac = file::JoinPath(FLAGS_test_tmpdir, "javac");
    CHECK_OK(file::SetContents(javac, "", file::Defaults()));
    CHECK_GE(chmod(javac.c_str(), 0755), 0);

    string path(getenv("PATH"));
    string test_tmpdir(getenv("TEST_TMPDIR"));
    path = test_tmpdir + ":" + path;
    setenv("PATH", path.c_str(), 1);
  }

  void TearDown() {
    file::RecursivelyDelete(build_root_, file::Defaults()).IgnoreError();
  }

  string build_root_;
  const std::unique_ptr<WorkspaceLayout> workspace_layout_;
};

TEST_F(WorkspaceLayoutTest, GetWorkspace) {
  // "" is returned when there's no workspace path.
  string cwd = "foo/bar";
  ASSERT_EQ("", workspace_layout_->GetWorkspace(cwd));
  ASSERT_FALSE(workspace_layout_->InWorkspace(cwd));

  cwd = build_root_;
  ASSERT_EQ(build_root_, workspace_layout_->GetWorkspace(cwd));
  ASSERT_TRUE(workspace_layout_->InWorkspace(build_root_));

  cwd = file::JoinPath(build_root_, "foo/bar");
  ASSERT_EQ(build_root_, workspace_layout_->GetWorkspace(cwd));
}

}  // namespace blaze
