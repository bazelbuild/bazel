// Copyright 2018 The Bazel Authors. All rights reserved.
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
#include <limits.h>
#include <unistd.h>

#include <algorithm>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/test/cpp/util/test_util.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

using std::string;

TEST(PathPosixTest, TestDirname) {
  // The Posix version of SplitPath (thus Dirname too, which is implemented on
  // top of it) is not aware of Windows paths.
  ASSERT_EQ("", Dirname(""));
  ASSERT_EQ("/", Dirname("/"));
  ASSERT_EQ("", Dirname("foo"));
  ASSERT_EQ("/", Dirname("/foo"));
  ASSERT_EQ("/foo", Dirname("/foo/"));
  ASSERT_EQ("foo", Dirname("foo/bar"));
  ASSERT_EQ("foo/bar", Dirname("foo/bar/baz"));
  ASSERT_EQ("", Dirname("\\foo"));
  ASSERT_EQ("", Dirname("\\foo\\"));
  ASSERT_EQ("", Dirname("foo\\bar"));
  ASSERT_EQ("", Dirname("foo\\bar\\baz"));
  ASSERT_EQ("foo\\bar", Dirname("foo\\bar/baz\\qux"));
  ASSERT_EQ("c:", Dirname("c:/"));
  ASSERT_EQ("", Dirname("c:\\"));
  ASSERT_EQ("c:", Dirname("c:/foo"));
  ASSERT_EQ("", Dirname("c:\\foo"));
  ASSERT_EQ("", Dirname("\\\\?\\c:\\"));
  ASSERT_EQ("", Dirname("\\\\?\\c:\\foo"));
}

TEST(PathPosixTest, TestBasename) {
  // The Posix version of SplitPath (thus Basename too, which is implemented on
  // top of it) is not aware of Windows paths.
  ASSERT_EQ("", Basename(""));
  ASSERT_EQ("", Basename("/"));
  ASSERT_EQ("foo", Basename("foo"));
  ASSERT_EQ("foo", Basename("/foo"));
  ASSERT_EQ("", Basename("/foo/"));
  ASSERT_EQ("bar", Basename("foo/bar"));
  ASSERT_EQ("baz", Basename("foo/bar/baz"));
  ASSERT_EQ("\\foo", Basename("\\foo"));
  ASSERT_EQ("\\foo\\", Basename("\\foo\\"));
  ASSERT_EQ("foo\\bar", Basename("foo\\bar"));
  ASSERT_EQ("foo\\bar\\baz", Basename("foo\\bar\\baz"));
  ASSERT_EQ("baz\\qux", Basename("foo\\bar/baz\\qux"));
  ASSERT_EQ("qux", Basename("qux"));
  ASSERT_EQ("", Basename("c:/"));
  ASSERT_EQ("c:\\", Basename("c:\\"));
  ASSERT_EQ("foo", Basename("c:/foo"));
  ASSERT_EQ("c:\\foo", Basename("c:\\foo"));
  ASSERT_EQ("\\\\?\\c:\\", Basename("\\\\?\\c:\\"));
  ASSERT_EQ("\\\\?\\c:\\foo", Basename("\\\\?\\c:\\foo"));
}

TEST(PathPosixTest, JoinPath) {
  std::string path = JoinPath("", "");
  ASSERT_EQ("", path);

  path = JoinPath("a", "b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a/", "b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a", "/b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a/", "/b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("/", "/");
  ASSERT_EQ("/", path);
}

TEST(PathPosixTest, GetCwd) {
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

TEST(PathPosixTest, IsAbsolute) {
  ASSERT_FALSE(IsAbsolute(""));
  ASSERT_TRUE(IsAbsolute("/"));
  ASSERT_TRUE(IsAbsolute("/foo"));
  ASSERT_FALSE(IsAbsolute("\\"));
  ASSERT_FALSE(IsAbsolute("\\foo"));
  ASSERT_FALSE(IsAbsolute("c:"));
  ASSERT_FALSE(IsAbsolute("c:/"));
  ASSERT_FALSE(IsAbsolute("c:\\"));
  ASSERT_FALSE(IsAbsolute("c:\\foo"));
  ASSERT_FALSE(IsAbsolute("\\\\?\\c:\\"));
  ASSERT_FALSE(IsAbsolute("\\\\?\\c:\\foo"));
}

TEST(PathPosixTest, IsRootDirectory) {
  ASSERT_FALSE(IsRootDirectory(""));
  ASSERT_TRUE(IsRootDirectory("/"));
  ASSERT_FALSE(IsRootDirectory("/foo"));
  ASSERT_FALSE(IsRootDirectory("\\"));
  ASSERT_FALSE(IsRootDirectory("\\foo"));
  ASSERT_FALSE(IsRootDirectory("c:"));
  ASSERT_FALSE(IsRootDirectory("c:/"));
  ASSERT_FALSE(IsRootDirectory("c:\\"));
  ASSERT_FALSE(IsRootDirectory("c:\\foo"));
  ASSERT_FALSE(IsRootDirectory("\\\\?\\c:\\"));
  ASSERT_FALSE(IsRootDirectory("\\\\?\\c:\\foo"));
}

TEST(PathPosixTest, IsDevNullTest) {
  ASSERT_TRUE(IsDevNull("/dev/null"));
  ASSERT_FALSE(IsDevNull("dev/null"));
  ASSERT_FALSE(IsDevNull("/dev/nul"));
  ASSERT_FALSE(IsDevNull("/dev/nulll"));
  ASSERT_FALSE(IsDevNull(NULL));
  ASSERT_FALSE(IsDevNull(""));
}

TEST(PathPosixTest, MakeAbsolute) {
  EXPECT_EQ(MakeAbsolute("/foo/bar"), "/foo/bar");
  EXPECT_EQ(MakeAbsolute("/foo/bar/"), "/foo/bar/");
  EXPECT_EQ(MakeAbsolute("foo"), blaze_util::GetCwd() + "/foo");

  EXPECT_EQ(MakeAbsolute("/dev/null"), "/dev/null");

  EXPECT_EQ(MakeAbsolute(""), "");
}

TEST(PathPosixTest, MakeAbsoluteAndResolveEnvvars) {
  // Check that Unix-style envvars are resolved.
  const std::string tmpdir = getenv("TEST_TMPDIR");
  const std::string expected_tmpdir_bar = ConvertPath(tmpdir + "/bar");
  setenv("PATH_POSIX_TEST_ENV", "${TEST_TMPDIR}", 1);

  // Using an existing environment variable
  EXPECT_EQ(expected_tmpdir_bar,
            MakeAbsoluteAndResolveEnvvars("${TEST_TMPDIR}/bar"));
  // Using an undefined environment variable (case-sensitive)
  EXPECT_EQ("/bar",
            MakeAbsoluteAndResolveEnvvars("${test_tmpdir}/bar"));

  // This style of variable is not supported
  EXPECT_EQ(JoinPath(GetCwd(), "$TEST_TMPDIR/bar"),
            MakeAbsoluteAndResolveEnvvars("$TEST_TMPDIR/bar"));

  // Only one layer of variables is expanded, we do not recurse
  EXPECT_EQ(JoinPath(GetCwd(), "${TEST_TMPDIR}/bar"),
            MakeAbsoluteAndResolveEnvvars("${PATH_POSIX_TEST_ENV}/bar"));

  // Check that Windows-style envvars are not resolved when not on Windows.
  EXPECT_EQ(MakeAbsoluteAndResolveEnvvars("%PATH%"),
            JoinPath(GetCwd(), "%PATH%"));
}

}  // namespace blaze_util
