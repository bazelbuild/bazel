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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>

#include "googletest/include/gtest/gtest.h"
#include "src/test/cpp/util/windows_test_util.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace blaze_util {

using std::wstring;

class WindowsTestUtilTest : public ::testing::Test {
 public:
  void SetUp() override { ::CreateDirectoryW(GetTestTmpDirW().c_str(), NULL); }
  void TearDown() override { DeleteAllUnder(GetTestTmpDirW()); }
};

TEST_F(WindowsTestUtilTest, TestGetTestTempDirW) {
  wstring actual = GetTestTmpDirW();
  ASSERT_EQ(actual.find(L":\\"), 1);
  ASSERT_EQ(actual.find(L"/"), wstring::npos);
}

TEST_F(WindowsTestUtilTest, TestCreateDummyFile) {
  wstring wtemp = GetTestTmpDirW();
  EXPECT_FALSE(wtemp.empty());
  wstring file1 = wstring(L"\\\\?\\") + wtemp + L"\\file1.txt";
  ASSERT_TRUE(CreateDummyFile(file1));
  DWORD attr = ::GetFileAttributesW(file1.c_str());
  ASSERT_NE(attr, INVALID_FILE_ATTRIBUTES);
}

TEST_F(WindowsTestUtilTest, TestDeleteAllUnder) {
  wstring wtemp = GetTestTmpDirW();
  EXPECT_FALSE(wtemp.empty());
  wstring dir1 = wstring(L"\\\\?\\") + wtemp + L"\\dir1";
  EXPECT_TRUE(::CreateDirectoryW(dir1.c_str(), NULL));
  EXPECT_TRUE(CreateDummyFile(dir1 + L"\\file1.txt"));
  wstring dir2 = dir1 + L"\\dir2";
  EXPECT_TRUE(::CreateDirectoryW(dir2.c_str(), NULL));
  EXPECT_TRUE(CreateDummyFile(dir2 + L"\\file2.txt"));
  ASSERT_TRUE(DeleteAllUnder(dir1));
}

}  // namespace blaze_util
