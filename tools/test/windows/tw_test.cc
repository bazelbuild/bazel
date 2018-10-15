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

// Tests for the Windows implementation of the test wrapper.

#include <windows.h>

#include <algorithm>

#include "gtest/gtest.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/native/windows/file.h"
#include "src/test/cpp/util/windows_test_util.h"
#include "tools/test/windows/tw.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace {

using bazel::tools::test_wrapper::FileInfo;
using bazel::tools::test_wrapper::testing::TestOnly_GetEnv;
using bazel::tools::test_wrapper::testing::TestOnly_GetFileListRelativeTo;

class TestWrapperWindowsTest : public ::testing::Test {
 public:
  void TearDown() override {
    blaze_util::DeleteAllUnder(blaze_util::GetTestTmpDirW());
  }
};

void GetTestTmpdir(std::wstring* result, int line) {
  EXPECT_TRUE(TestOnly_GetEnv(L"TEST_TMPDIR", result))
      << __FILE__ << "(" << line << "): assertion failed here";
  ASSERT_GT(result->size(), 0)
      << __FILE__ << "(" << line << "): assertion failed here";
  std::replace(result->begin(), result->end(), L'/', L'\\');
  if (!bazel::windows::HasUncPrefix(result->c_str())) {
    *result = L"\\\\?\\" + *result;
  }
}

void CreateJunction(const std::wstring& name, const std::wstring& target,
                    int line) {
  std::wstring wname;
  std::wstring wtarget;
  EXPECT_TRUE(blaze_util::AsWindowsPath(name, &wname, nullptr))
      << __FILE__ << "(" << line << "): assertion failed here";
  EXPECT_TRUE(blaze_util::AsWindowsPath(target, &wtarget, nullptr))
      << __FILE__ << "(" << line << "): assertion failed here";
  EXPECT_EQ(bazel::windows::CreateJunction(wname, wtarget, nullptr),
            bazel::windows::CreateJunctionResult::kSuccess)
      << __FILE__ << "(" << line << "): assertion failed here";
}

void CompareFileInfos(std::vector<FileInfo> actual,
                      std::vector<FileInfo> expected, int line) {
  ASSERT_EQ(actual.size(), expected.size())
      << __FILE__ << "(" << line << "): assertion failed here";
  std::sort(actual.begin(), actual.end(),
            [](const FileInfo& a, const FileInfo& b) {
              return a.rel_path > b.rel_path;
            });
  std::sort(expected.begin(), expected.end(),
            [](const FileInfo& a, const FileInfo& b) {
              return a.rel_path > b.rel_path;
            });
  for (std::vector<FileInfo>::size_type i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].rel_path, expected[i].rel_path)
        << __FILE__ << "(" << line << "): assertion failed here; index: " << i;
    ASSERT_EQ(actual[i].size, expected[i].size)
        << __FILE__ << "(" << line << "): assertion failed here; index: " << i;
  }
}

#define GET_TEST_TMPDIR(result) GetTestTmpdir(result, __LINE__)
#define CREATE_JUNCTION(name, target) CreateJunction(name, target, __LINE__)
#define COMPARE_FILE_INFOS(actual, expected) \
  CompareFileInfos(actual, expected, __LINE__)

#define TOSTRING1(x) #x
#define TOSTRING(x) TOSTRING1(x)
#define TOWSTRING1(x) L##x
#define TOWSTRING(x) TOWSTRING1(x)
#define WLINE TOWSTRING(TOSTRING(__LINE__))

TEST_F(TestWrapperWindowsTest, TestGetFileListRelativeTo) {
  std::wstring tmpdir;
  GET_TEST_TMPDIR(&tmpdir);

  // Create a directory structure to parse.
  std::wstring root = tmpdir + L"\\tmp" + WLINE;
  EXPECT_TRUE(CreateDirectoryW(root.c_str(), NULL));
  EXPECT_TRUE(CreateDirectoryW((root + L"\\foo").c_str(), NULL));
  EXPECT_TRUE(CreateDirectoryW((root + L"\\foo\\sub").c_str(), NULL));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\foo\\sub\\file1", ""));
  EXPECT_TRUE(
      blaze_util::CreateDummyFile(root + L"\\foo\\sub\\file2", "hello"));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\foo\\file1", "foo"));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\foo\\file2", "foobar"));
  CREATE_JUNCTION(root + L"\\foo\\junc", root + L"\\foo\\sub");

  // Assert traversal of "root" -- should include all files, and also traverse
  // the junction.
  std::vector<FileInfo> actual;
  ASSERT_TRUE(TestOnly_GetFileListRelativeTo(root, &actual));

  std::vector<FileInfo> expected = {
      {L"foo\\sub\\file1", 0},  {L"foo\\sub\\file2", 5},
      {L"foo\\file1", 3},       {L"foo\\file2", 6},
      {L"foo\\junc\\file1", 0}, {L"foo\\junc\\file2", 5}};
  COMPARE_FILE_INFOS(actual, expected);

  // Assert traversal of "foo" -- should include all files, but now with paths
  // relative to "foo".
  actual.clear();
  ASSERT_TRUE(
      TestOnly_GetFileListRelativeTo((root + L"\\foo").c_str(), &actual));

  expected = {{L"sub\\file1", 0}, {L"sub\\file2", 5},  {L"file1", 3},
              {L"file2", 6},      {L"junc\\file1", 0}, {L"junc\\file2", 5}};
  COMPARE_FILE_INFOS(actual, expected);
}

}  // namespace
