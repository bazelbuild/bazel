// Copyright 2017 The Bazel Authors. All rights reserved.
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
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <memory>  // unique_ptr
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "src/main/native/windows/file.h"
#include "src/test/cpp/util/windows_test_util.h"

#if !defined(COMPILER_MSVC) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(COMPILER_MSVC) && !defined(__CYGWIN__)

namespace bazel {
namespace windows {

using blaze_util::DeleteAllUnder;
using blaze_util::GetTestTmpDirW;
using std::unique_ptr;
using std::wstring;

static const wstring kUncPrefix = wstring(L"\\\\?\\");

class WindowsFileOperationsTest : public ::testing::Test {
 public:
  void TearDown() override { DeleteAllUnder(GetTestTmpDirW()); }
};

TEST_F(WindowsFileOperationsTest, TestCreateJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring target(tmp + L"\\junc_target");
  EXPECT_TRUE(::CreateDirectoryW(target.c_str(), NULL));
  wstring file1(target + L"\\foo");
  EXPECT_TRUE(blaze_util::CreateDummyFile(file1));

  EXPECT_EQ(IS_JUNCTION_NO, IsJunctionOrDirectorySymlink(target.c_str()));
  EXPECT_NE(INVALID_FILE_ATTRIBUTES, ::GetFileAttributesW(file1.c_str()));

  wstring name(tmp + L"\\junc_name");

  // Create junctions from all combinations of UNC-prefixed or non-prefixed name
  // and target paths.
  ASSERT_EQ(L"", CreateJunction(name + L"1", target));
  ASSERT_EQ(L"", CreateJunction(name + L"2", target.substr(4)));
  ASSERT_EQ(L"", CreateJunction(name.substr(4) + L"3", target));
  ASSERT_EQ(L"", CreateJunction(name.substr(4) + L"4", target.substr(4)));

  // Assert creation of the junctions.
  ASSERT_EQ(IS_JUNCTION_YES,
            IsJunctionOrDirectorySymlink((name + L"1").c_str()));
  ASSERT_EQ(IS_JUNCTION_YES,
            IsJunctionOrDirectorySymlink((name + L"2").c_str()));
  ASSERT_EQ(IS_JUNCTION_YES,
            IsJunctionOrDirectorySymlink((name + L"3").c_str()));
  ASSERT_EQ(IS_JUNCTION_YES,
            IsJunctionOrDirectorySymlink((name + L"4").c_str()));

  // Assert that the file is visible under all junctions.
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"1\\foo").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"2\\foo").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"3\\foo").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"4\\foo").c_str()));

  // Assert that no other file exists under the junctions.
  wstring file2(target + L"\\bar");
  ASSERT_EQ(INVALID_FILE_ATTRIBUTES, ::GetFileAttributesW(file2.c_str()));
  ASSERT_EQ(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"1\\bar").c_str()));
  ASSERT_EQ(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"2\\bar").c_str()));
  ASSERT_EQ(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"3\\bar").c_str()));
  ASSERT_EQ(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"4\\bar").c_str()));

  // Create a new file.
  EXPECT_TRUE(blaze_util::CreateDummyFile(file2));
  EXPECT_NE(INVALID_FILE_ATTRIBUTES, ::GetFileAttributesW(file2.c_str()));

  // Assert that the newly created file appears under all junctions.
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"1\\bar").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"2\\bar").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"3\\bar").c_str()));
  ASSERT_NE(INVALID_FILE_ATTRIBUTES,
            ::GetFileAttributesW((name + L"4\\bar").c_str()));
}

}  // namespace windows
}  // namespace bazel
