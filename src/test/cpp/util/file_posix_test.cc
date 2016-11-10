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
#include <fcntl.h>
#include <unistd.h>

#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

namespace blaze_util {

TEST(FilePosixTest, Which) {
  ASSERT_EQ("", Which(""));
  ASSERT_EQ("", Which("foo"));
  ASSERT_EQ("", Which("/"));

  // /usr/bin/yes exists on Linux, Darwin, and MSYS, but "which yes" does not
  // always return that (if $PATH is different).
  string actual = Which("yes");
  // Assert that it's an absolute path
  ASSERT_EQ(0, actual.find("/"));
  // Assert that it ends with /yes, we cannot assume more than that.
  ASSERT_EQ(actual.size() - string("/yes").size(), actual.rfind("/yes"));
}

TEST(FilePosixTest, PathExists) {
  ASSERT_FALSE(PathExists("/this/should/not/exist/mkay"));
  ASSERT_FALSE(PathExists("non.existent"));
  ASSERT_FALSE(PathExists(""));

  // /usr/bin/yes exists on Linux, Darwin, and MSYS
  ASSERT_TRUE(PathExists("/"));
  ASSERT_TRUE(PathExists("/usr"));
  ASSERT_TRUE(PathExists("/usr/"));
  ASSERT_TRUE(PathExists("/usr/bin/yes"));
}

TEST(FilePosixTest, CanAccess) {
  for (int i = 0; i < 8; ++i) {
    ASSERT_FALSE(CanAccess("/this/should/not/exist/mkay", i & 1, i & 2, i & 4));
    ASSERT_FALSE(CanAccess("non.existent", i & 1, i & 2, i & 4));
  }

  for (int i = 0; i < 4; ++i) {
    // /usr/bin/yes exists on Linux, Darwin, and MSYS
    ASSERT_TRUE(CanAccess("/", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr/", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr/bin/yes", i & 1, false, i & 2));
  }

  char* tmpdir_cstr = getenv("TEST_TMPDIR");
  ASSERT_FALSE(tmpdir_cstr == NULL);

  string tmpdir(tmpdir_cstr);
  ASSERT_NE("", tmpdir);

  string mock_file = tmpdir + (tmpdir.back() == '/' ? "" : "/") +
                     "FilePosixTest.CanAccess.mock_file";
  int fd = open(mock_file.c_str(), O_CREAT, 0500);
  ASSERT_GT(fd, 0);
  close(fd);

  // Sanity check: assert that we successfully created the file with the given
  // permissions.
  ASSERT_EQ(0, access(mock_file.c_str(), R_OK | X_OK));
  ASSERT_NE(0, access(mock_file.c_str(), R_OK | W_OK | X_OK));

  // Actual assertion
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(CanAccess(mock_file, i & 1, false, i & 2));
    ASSERT_FALSE(CanAccess(mock_file, i & 1, true, i & 2));
  }
}

TEST(FilePosixTest, GetCwd) {
  char cwdbuf[PATH_MAX];
  ASSERT_EQ(cwdbuf, getcwd(cwdbuf, PATH_MAX));

  // Assert that GetCwd() and getcwd() return the same value.
  string cwd(cwdbuf);
  ASSERT_EQ(cwd, blaze_util::GetCwd());

  // Change to a different directory.
  ASSERT_EQ(0, chdir("/usr"));

  // Assert that GetCwd() returns the new CWD.
  ASSERT_EQ(string("/usr"), blaze_util::GetCwd());

  ASSERT_EQ(0, chdir(cwd.c_str()));
  ASSERT_EQ(cwd, blaze_util::GetCwd());
}

TEST(FilePosixTest, ChangeDirectory) {
  // Retrieve the current working directory.
  char old_wd[PATH_MAX];
  ASSERT_EQ(old_wd, getcwd(old_wd, PATH_MAX));

  // Change to a different directory and assert it was successful.
  ASSERT_FALSE(blaze_util::ChangeDirectory("/non/existent/path"));
  ASSERT_TRUE(blaze_util::ChangeDirectory("/usr"));
  char new_wd[PATH_MAX];
  ASSERT_EQ(new_wd, getcwd(new_wd, PATH_MAX));
  ASSERT_EQ(string("/usr"), string(new_wd));

  // Change back to the original CWD.
  ASSERT_TRUE(blaze_util::ChangeDirectory(old_wd));
  ASSERT_EQ(new_wd, getcwd(new_wd, PATH_MAX));
  ASSERT_EQ(string(old_wd), string(new_wd));
}

}  // namespace blaze_util
