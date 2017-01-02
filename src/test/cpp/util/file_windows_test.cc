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
#include <string.h>
#include <windows.h>  // SetEnvironmentVariableA

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

#if !defined(COMPILER_MSVC) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(COMPILER_MSVC) && !defined(__CYGWIN__)

namespace blaze_util {

void ReinitMsysRootForTesting();  // defined in file_windows.cc

TEST(FileTest, TestDirname) {
  ASSERT_EQ("", Dirname(""));
  ASSERT_EQ("/", Dirname("/"));
  ASSERT_EQ("", Dirname("foo"));
  ASSERT_EQ("/", Dirname("/foo"));
  ASSERT_EQ("/foo", Dirname("/foo/"));
  ASSERT_EQ("foo", Dirname("foo/bar"));
  ASSERT_EQ("foo/bar", Dirname("foo/bar/baz"));
  ASSERT_EQ("\\", Dirname("\\foo"));
  ASSERT_EQ("\\foo", Dirname("\\foo\\"));
  ASSERT_EQ("foo", Dirname("foo\\bar"));
  ASSERT_EQ("foo\\bar", Dirname("foo\\bar\\baz"));
  ASSERT_EQ("foo\\bar/baz", Dirname("foo\\bar/baz\\qux"));
  ASSERT_EQ("c:/", Dirname("c:/"));
  ASSERT_EQ("c:\\", Dirname("c:\\"));
  ASSERT_EQ("c:/", Dirname("c:/foo"));
  ASSERT_EQ("c:\\", Dirname("c:\\foo"));
  ASSERT_EQ("\\\\?\\c:\\", Dirname("\\\\?\\c:\\"));
  ASSERT_EQ("\\\\?\\c:\\", Dirname("\\\\?\\c:\\foo"));
}

TEST(FileTest, TestBasename) {
  ASSERT_EQ("", Basename(""));
  ASSERT_EQ("", Basename("/"));
  ASSERT_EQ("foo", Basename("foo"));
  ASSERT_EQ("foo", Basename("/foo"));
  ASSERT_EQ("", Basename("/foo/"));
  ASSERT_EQ("bar", Basename("foo/bar"));
  ASSERT_EQ("baz", Basename("foo/bar/baz"));
  ASSERT_EQ("foo", Basename("\\foo"));
  ASSERT_EQ("", Basename("\\foo\\"));
  ASSERT_EQ("bar", Basename("foo\\bar"));
  ASSERT_EQ("baz", Basename("foo\\bar\\baz"));
  ASSERT_EQ("qux", Basename("foo\\bar/baz\\qux"));
  ASSERT_EQ("", Basename("c:/"));
  ASSERT_EQ("", Basename("c:\\"));
  ASSERT_EQ("foo", Basename("c:/foo"));
  ASSERT_EQ("foo", Basename("c:\\foo"));
  ASSERT_EQ("", Basename("\\\\?\\c:\\"));
  ASSERT_EQ("foo", Basename("\\\\?\\c:\\foo"));
}

TEST(FileTest, IsAbsolute) {
  ASSERT_FALSE(IsAbsolute(""));
  ASSERT_TRUE(IsAbsolute("/"));
  ASSERT_TRUE(IsAbsolute("/foo"));
  ASSERT_TRUE(IsAbsolute("\\"));
  ASSERT_TRUE(IsAbsolute("\\foo"));
  ASSERT_FALSE(IsAbsolute("c:"));
  ASSERT_TRUE(IsAbsolute("c:/"));
  ASSERT_TRUE(IsAbsolute("c:\\"));
  ASSERT_TRUE(IsAbsolute("c:\\foo"));
  ASSERT_TRUE(IsAbsolute("\\\\?\\c:\\"));
  ASSERT_TRUE(IsAbsolute("\\\\?\\c:\\foo"));
}

TEST(FileTest, IsRootDirectory) {
  ASSERT_FALSE(IsRootDirectory(""));
  ASSERT_TRUE(IsRootDirectory("/"));
  ASSERT_FALSE(IsRootDirectory("/foo"));
  ASSERT_TRUE(IsRootDirectory("\\"));
  ASSERT_FALSE(IsRootDirectory("\\foo"));
  ASSERT_FALSE(IsRootDirectory("c:"));
  ASSERT_TRUE(IsRootDirectory("c:/"));
  ASSERT_TRUE(IsRootDirectory("c:\\"));
  ASSERT_FALSE(IsRootDirectory("c:\\foo"));
  ASSERT_TRUE(IsRootDirectory("\\\\?\\c:\\"));
  ASSERT_FALSE(IsRootDirectory("\\\\?\\c:\\foo"));
}

TEST(FileTest, TestAsWindowsPath) {
  SetEnvironmentVariableA("BAZEL_SH", "c:\\msys\\some\\long\\path\\bash.exe");
  ReinitMsysRootForTesting();
  std::wstring actual;

  ASSERT_TRUE(AsWindowsPath("", &actual));
  ASSERT_EQ(std::wstring(L""), actual);

  ASSERT_TRUE(AsWindowsPath("", &actual));
  ASSERT_EQ(std::wstring(L""), actual);

  ASSERT_TRUE(AsWindowsPath("foo/bar", &actual));
  ASSERT_EQ(std::wstring(L"foo\\bar"), actual);

  ASSERT_TRUE(AsWindowsPath("/c", &actual));
  ASSERT_EQ(std::wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("/c/", &actual));
  ASSERT_EQ(std::wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("/c/blah", &actual));
  ASSERT_EQ(std::wstring(L"c:\\blah"), actual);

  ASSERT_TRUE(AsWindowsPath("/d/progra~1/micros~1", &actual));
  ASSERT_EQ(std::wstring(L"d:\\progra~1\\micros~1"), actual);

  ASSERT_TRUE(AsWindowsPath("/foo", &actual));
  ASSERT_EQ(std::wstring(L"c:\\msys\\foo"), actual);

  std::wstring wlongpath(L"dummy_long_path\\");
  std::string longpath("dummy_long_path/");
  while (longpath.size() <= MAX_PATH) {
    wlongpath += wlongpath;
    longpath += longpath;
  }
  wlongpath = std::wstring(L"c:\\") + wlongpath;
  longpath = std::string("/c/") + longpath;
  ASSERT_TRUE(AsWindowsPath(longpath, &actual));
  ASSERT_EQ(wlongpath, actual);
}

TEST(FileTest, TestMsysRootRetrieval) {
  std::wstring actual;

  SetEnvironmentVariableA("BAZEL_SH", "c:/foo/msys/bar/qux.exe");
  ReinitMsysRootForTesting();
  ASSERT_TRUE(AsWindowsPath("/blah", &actual));
  ASSERT_EQ(std::wstring(L"c:\\foo\\msys\\blah"), actual);

  SetEnvironmentVariableA("BAZEL_SH", "c:/foo/MSYS64/bar/qux.exe");
  ReinitMsysRootForTesting();
  ASSERT_TRUE(AsWindowsPath("/blah", &actual));
  ASSERT_EQ(std::wstring(L"c:\\foo\\msys64\\blah"), actual);

  SetEnvironmentVariableA("BAZEL_SH", "c:/qux.exe");
  ReinitMsysRootForTesting();
  ASSERT_FALSE(AsWindowsPath("/blah", &actual));

  SetEnvironmentVariableA("BAZEL_SH", nullptr);
  ReinitMsysRootForTesting();
}

}  // namespace blaze_util
