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

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace bazel {
namespace windows {

#define TOSTRING1(x) #x
#define TOSTRING(x) TOSTRING1(x)
#define TOWSTRING1(x) L##x
#define TOWSTRING(x) TOWSTRING1(x)
#define WLINE TOWSTRING(TOSTRING(__LINE__))

using blaze_util::DeleteAllUnder;
using blaze_util::GetTestTmpDirW;
using std::unique_ptr;
using std::wstring;

static const wstring kUncPrefix = wstring(L"\\\\?\\");

class WindowsFileOperationsTest : public ::testing::Test {
 public:
  void TearDown() override { DeleteAllUnder(GetTestTmpDirW()); }
};

TEST_F(WindowsFileOperationsTest, TestIsAbsoluteWindowsStylePath) {
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L""));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"NUL"));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"nul"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:/"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:/"));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"c:\\"));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:\\foo/bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\foo/bar"));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"c:\\foo\\bar"));
  EXPECT_TRUE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\foo\\bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"foo"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"foo\\bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:\\foo\\."));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\foo\\."));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:\\foo\\.\\bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\foo\\.\\bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"c:\\foo\\..\\bar"));
  EXPECT_FALSE(IsAbsoluteNormalizedWindowsPath(L"\\\\?\\c:\\foo\\..\\bar"));
}

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
  ASSERT_EQ(CreateJunction(name + L"1", target, nullptr),
            CreateJunctionResult::kSuccess);
  ASSERT_EQ(CreateJunction(name + L"2", target.substr(4), nullptr),
            CreateJunctionResult::kSuccess);
  ASSERT_EQ(CreateJunction(name.substr(4) + L"3", target, nullptr),
            CreateJunctionResult::kSuccess);
  ASSERT_EQ(CreateJunction(name.substr(4) + L"4", target.substr(4), nullptr),
            CreateJunctionResult::kSuccess);

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

TEST_F(WindowsFileOperationsTest, TestCanCreateNonDanglingJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanCreateDanglingJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCreateJunctionChecksExistingJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);

  ASSERT_EQ(CreateJunction(name, target + WLINE, nullptr),
            CreateJunctionResult::kAlreadyExistsWithDifferentTarget);
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCannotCreateJunctionFromEmptyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), NULL));
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest,
       TestCannotCreateJunctionFromNonEmptyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), NULL));
  EXPECT_TRUE(blaze_util::CreateDummyFile(name + L"\\hello.txt"));
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest, TestCannotCreateJunctionFromExistingFile) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(blaze_util::CreateDummyFile(name));
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest, TestCannotCreateButCanCheckIfNameIsBusy) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), NULL));
  HANDLE h = CreateFileW(
      name.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = CreateJunction(name, target, nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest, TestCanCreateJunctionIfTargetIsBusy) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  HANDLE h = CreateFileW(target.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = CreateJunction(name, target, nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, CreateJunctionResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingFile) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\file" WLINE;
  EXPECT_TRUE(blaze_util::CreateDummyFile(path));
  ASSERT_EQ(DeletePath(path.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\dir" WLINE;
  EXPECT_TRUE(CreateDirectoryW(path.c_str(), NULL));
  ASSERT_EQ(DeletePath(path.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  ASSERT_EQ(DeletePath(name.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingJunctionWithoutTarget) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  EXPECT_TRUE(RemoveDirectoryW(target.c_str()));
  // The junction still exists, its target does not.
  EXPECT_NE(GetFileAttributesW(name.c_str()), INVALID_FILE_ATTRIBUTES);
  EXPECT_EQ(GetFileAttributesW(target.c_str()), INVALID_FILE_ATTRIBUTES);
  // We can delete the dangling junction.
  ASSERT_EQ(DeletePath(name.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteNonExistentPath) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\dummy" WLINE;
  EXPECT_EQ(GetFileAttributesW(path.c_str()), INVALID_FILE_ATTRIBUTES);
  ASSERT_EQ(DeletePath(path.c_str(), nullptr), DeletePathResult::kDoesNotExist);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeletePathWhereParentIsFile) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring parent = tmp + L"\\file" WLINE;
  wstring child = parent + L"\\file" WLINE;
  EXPECT_TRUE(blaze_util::CreateDummyFile(parent));
  ASSERT_EQ(DeletePath(child.c_str(), nullptr),
            DeletePathResult::kDoesNotExist);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteNonEmptyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring parent = tmp + L"\\dir" WLINE;
  wstring child = parent + L"\\file" WLINE;
  EXPECT_TRUE(CreateDirectoryW(parent.c_str(), NULL));
  EXPECT_TRUE(blaze_util::CreateDummyFile(child));
  ASSERT_EQ(DeletePath(parent.c_str(), nullptr),
            DeletePathResult::kDirectoryNotEmpty);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyFile) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\file" WLINE;
  EXPECT_TRUE(blaze_util::CreateDummyFile(path));
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(path.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\dir" WLINE;
  EXPECT_TRUE(CreateDirectoryW(path.c_str(), NULL));
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(path.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  // Open the junction itself (do not follow symlinks).
  HANDLE h = CreateFileW(
      name.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(name.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteJunctionWhoseTargetIsBusy) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), NULL));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  // Open the junction's target (follow symlinks).
  HANDLE h = CreateFileW(target.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS, NULL);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(name.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kSuccess);
}

#undef TOSTRING1
#undef TOSTRING
#undef TOWSTRING1
#undef TOWSTRING
#undef WLINE

}  // namespace windows
}  // namespace bazel
