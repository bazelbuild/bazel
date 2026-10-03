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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "src/main/native/windows/file.h"

#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <memory>  // unique_ptr
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
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
  EXPECT_TRUE(::CreateDirectoryW(target.c_str(), nullptr));
  wstring file1(target + L"\\foo");
  EXPECT_TRUE(blaze_util::CreateDummyFile(file1));

  bool is_link = true;
  EXPECT_EQ(IsSymlinkOrJunctionResult::kSuccess,
            IsSymlinkOrJunction(target.c_str(), &is_link, nullptr));
  EXPECT_FALSE(is_link);
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
  is_link = false;
  ASSERT_EQ(IsSymlinkOrJunctionResult::kSuccess,
            IsSymlinkOrJunction((name + L"1").c_str(), &is_link, nullptr));
  ASSERT_TRUE(is_link);
  is_link = false;
  ASSERT_EQ(IsSymlinkOrJunctionResult::kSuccess,
            IsSymlinkOrJunction((name + L"2").c_str(), &is_link, nullptr));
  ASSERT_TRUE(is_link);
  is_link = false;
  ASSERT_EQ(IsSymlinkOrJunctionResult::kSuccess,
            IsSymlinkOrJunction((name + L"3").c_str(), &is_link, nullptr));
  ASSERT_TRUE(is_link);
  is_link = false;
  ASSERT_EQ(IsSymlinkOrJunctionResult::kSuccess,
            IsSymlinkOrJunction((name + L"4").c_str(), &is_link, nullptr));
  ASSERT_TRUE(is_link);

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
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
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
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), nullptr));
  ASSERT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest,
       TestCannotCreateJunctionFromNonEmptyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), nullptr));
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
  EXPECT_TRUE(CreateDirectoryW(name.c_str(), nullptr));
  HANDLE h = CreateFileW(
      name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = CreateJunction(name, target, nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, CreateJunctionResult::kAlreadyExistsButNotJunction);
}

TEST_F(WindowsFileOperationsTest, TestCanCreateJunctionIfTargetIsBusy) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
  HANDLE h = CreateFileW(target.c_str(), GENERIC_WRITE, 0, nullptr,
                         OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
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
  EXPECT_TRUE(CreateDirectoryW(path.c_str(), nullptr));
  ASSERT_EQ(DeletePath(path.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  ASSERT_EQ(DeletePath(name.c_str(), nullptr), DeletePathResult::kSuccess);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteExistingJunctionWithoutTarget) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
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
  EXPECT_TRUE(CreateDirectoryW(parent.c_str(), nullptr));
  EXPECT_TRUE(blaze_util::CreateDummyFile(child));
  ASSERT_EQ(DeletePath(parent.c_str(), nullptr),
            DeletePathResult::kDirectoryNotEmpty);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyFile) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\file" WLINE;
  EXPECT_TRUE(blaze_util::CreateDummyFile(path));
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(path.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyDirectory) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring path = tmp + L"\\dir" WLINE;
  EXPECT_TRUE(CreateDirectoryW(path.c_str(), nullptr));
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(path.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCannotDeleteBusyJunction) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  // Open the junction itself (do not follow symlinks).
  HANDLE h = CreateFileW(
      name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(name.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kAccessDenied);
}

TEST_F(WindowsFileOperationsTest, TestCanDeleteJunctionWhoseTargetIsBusy) {
  wstring tmp(kUncPrefix + GetTestTmpDirW());
  wstring name = tmp + L"\\junc" WLINE;
  wstring target = tmp + L"\\target" WLINE;
  EXPECT_TRUE(CreateDirectoryW(target.c_str(), nullptr));
  EXPECT_EQ(CreateJunction(name, target, nullptr),
            CreateJunctionResult::kSuccess);
  // Open the junction's target (follow symlinks).
  HANDLE h = CreateFileW(target.c_str(), GENERIC_WRITE, 0, nullptr,
                         OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  EXPECT_NE(h, INVALID_HANDLE_VALUE);
  int actual = DeletePath(name.c_str(), nullptr);
  CloseHandle(h);
  ASSERT_EQ(actual, DeletePathResult::kSuccess);
}

// Unmounts a volume mount point when it goes out of scope, so that a failing
// assertion cannot leave a volume mounted in the test's temp directory.
class ScopedVolumeMountPoint {
 public:
  explicit ScopedVolumeMountPoint(const wstring& path) : path_(path) {}
  ~ScopedVolumeMountPoint() { ::DeleteVolumeMountPointW(path_.c_str()); }

 private:
  const wstring path_;
};

// End-to-end check on a real volume mount point ("mounted folder"). Creating
// one needs administrator rights, so this test skips without them;
// TestInterpretReparseDataOfVolumeMountPoint covers the same logic on every
// run. To reproduce by hand, from an elevated prompt:
//   mkdir C:\mp
//   mountvol C:\mp\ \\?\Volume{GUID}\    (a volume name from `mountvol`)
//   fsutil reparsepoint query C:\mp      (Mount Point, "\??\Volume{GUID}\")
//   mountvol C:\mp\ /D                   (removes the mount point again)
// or with a fresh virtual disk: `diskpart` with
//   create vdisk file=C:\mp.vhdx maximum=512 type=expandable
//   attach vdisk / create partition primary / format fs=ntfs quick
//   assign mount=C:\mp
// A Bazel workspace in C:\mp then failed with errors like "no such package
// '...': BUILD file not found", because Bazel followed the mount point as if it
// were a junction to the relative path "Volume{GUID}".
TEST_F(WindowsFileOperationsTest, TestVolumeMountPointIsNotALink) {
  wstring tmp(GetTestTmpDirW());
  wstring mount_point = tmp + L"\\mnt" WLINE;
  wstring junction = tmp + L"\\junc" WLINE;
  wstring junction_target = tmp + L"\\target" WLINE;
  ASSERT_TRUE(::CreateDirectoryW(mount_point.c_str(), nullptr));
  ASSERT_TRUE(::CreateDirectoryW(junction_target.c_str(), nullptr));
  ASSERT_EQ(CreateJunction(junction, junction_target, nullptr),
            CreateJunctionResult::kSuccess);

  // Mount the volume that holds the temp directory onto `mount_point`.
  WCHAR volume_root[MAX_PATH];
  ASSERT_TRUE(::GetVolumePathNameW(tmp.c_str(), volume_root, MAX_PATH));
  WCHAR volume_name[MAX_PATH];
  ASSERT_TRUE(
      ::GetVolumeNameForVolumeMountPointW(volume_root, volume_name, MAX_PATH));
  // SetVolumeMountPointW requires a trailing backslash.
  if (!::SetVolumeMountPointW((mount_point + L"\\").c_str(), volume_name)) {
    DWORD err = GetLastError();
    if (err == ERROR_ACCESS_DENIED || err == ERROR_PRIVILEGE_NOT_HELD) {
      GTEST_SKIP() << "Creating a volume mount point requires administrator "
                      "privileges";
    }
    FAIL() << "SetVolumeMountPointW failed with error " << err;
  }
  ScopedVolumeMountPoint unmount(mount_point + L"\\");

  // The mount point is reported as a directory, not as a link, with or without
  // the "\\?\" prefix.
  for (const wstring& path : {mount_point, kUncPrefix + mount_point}) {
    bool is_link = true;
    EXPECT_EQ(IsSymlinkOrJunction(path.c_str(), &is_link, nullptr),
              IsSymlinkOrJunctionResult::kSuccess);
    EXPECT_FALSE(is_link);
    wstring target;
    EXPECT_EQ(ReadSymlinkOrJunction(path, &target, nullptr),
              ReadSymlinkOrJunctionResult::kNotALink);
    EXPECT_TRUE(target.empty());
  }

  // An ordinary junction is still a link.
  bool is_link = false;
  EXPECT_EQ(IsSymlinkOrJunction(junction.c_str(), &is_link, nullptr),
            IsSymlinkOrJunctionResult::kSuccess);
  EXPECT_TRUE(is_link);
  wstring target;
  EXPECT_EQ(ReadSymlinkOrJunction(junction, &target, nullptr),
            ReadSymlinkOrJunctionResult::kSuccess);
  EXPECT_EQ(target, L"\\??\\" + junction_target);
}

#undef TOSTRING1
#undef TOSTRING
#undef TOWSTRING1
#undef TOWSTRING
#undef WLINE

TEST(FileTests, TestNormalize) {
#define ASSERT_NORMALIZE(x, y) EXPECT_EQ(Normalize(x), y);
  ASSERT_NORMALIZE("", "");
  ASSERT_NORMALIZE("a", "a");
  ASSERT_NORMALIZE("foo/bar", "foo\\bar");
  ASSERT_NORMALIZE("foo/../bar", "bar");
  ASSERT_NORMALIZE("a/", "a");
  ASSERT_NORMALIZE("foo", "foo");
  ASSERT_NORMALIZE("foo/", "foo");
  ASSERT_NORMALIZE(".", ".");
  ASSERT_NORMALIZE("./", ".");
  ASSERT_NORMALIZE("..", "..");
  ASSERT_NORMALIZE("../", "..");
  ASSERT_NORMALIZE("./..", "..");
  ASSERT_NORMALIZE("./../", "..");
  ASSERT_NORMALIZE("../.", "..");
  ASSERT_NORMALIZE(".././", "..");
  ASSERT_NORMALIZE("...", "...");
  ASSERT_NORMALIZE(".../", "...");
  ASSERT_NORMALIZE("a/", "a");
  ASSERT_NORMALIZE(".a", ".a");
  ASSERT_NORMALIZE("..a", "..a");
  ASSERT_NORMALIZE("...a", "...a");
  ASSERT_NORMALIZE("./a", "a");
  ASSERT_NORMALIZE("././a", "a");
  ASSERT_NORMALIZE("./../a", "..\\a");
  ASSERT_NORMALIZE(".././a", "..\\a");
  ASSERT_NORMALIZE("../../a", "..\\..\\a");
  ASSERT_NORMALIZE("../.../a", "..\\...\\a");
  ASSERT_NORMALIZE(".../../a", "a");
  ASSERT_NORMALIZE("a/..", "");
  ASSERT_NORMALIZE("a/../", "");
  ASSERT_NORMALIZE("a/./../", "");

  ASSERT_NORMALIZE("c:/", "c:\\");
  ASSERT_NORMALIZE("c:/a", "c:\\a");
  ASSERT_NORMALIZE("c:/foo/bar", "c:\\foo\\bar");
  ASSERT_NORMALIZE("c:/foo/../bar", "c:\\bar");
  ASSERT_NORMALIZE("d:/a/", "d:\\a");
  ASSERT_NORMALIZE("D:/foo", "D:\\foo");
  ASSERT_NORMALIZE("c:/foo/", "c:\\foo");
  ASSERT_NORMALIZE("c:/.", "c:\\");
  ASSERT_NORMALIZE("c:/./", "c:\\");
  ASSERT_NORMALIZE("c:/..", "c:\\");
  ASSERT_NORMALIZE("c:/../", "c:\\");
  ASSERT_NORMALIZE("c:/./..", "c:\\");
  ASSERT_NORMALIZE("c:/./../", "c:\\");
  ASSERT_NORMALIZE("c:/../.", "c:\\");
  ASSERT_NORMALIZE("c:/.././", "c:\\");
  ASSERT_NORMALIZE("c:/...", "c:\\...");
  ASSERT_NORMALIZE("c:/.../", "c:\\...");
  ASSERT_NORMALIZE("c:/.a", "c:\\.a");
  ASSERT_NORMALIZE("c:/..a", "c:\\..a");
  ASSERT_NORMALIZE("c:/...a", "c:\\...a");
  ASSERT_NORMALIZE("c:/./a", "c:\\a");
  ASSERT_NORMALIZE("c:/././a", "c:\\a");
  ASSERT_NORMALIZE("c:/./../a", "c:\\a");
  ASSERT_NORMALIZE("c:/.././a", "c:\\a");
  ASSERT_NORMALIZE("c:/../../a", "c:\\a");
  ASSERT_NORMALIZE("c:/../.../a", "c:\\...\\a");
  ASSERT_NORMALIZE("c:/.../../a", "c:\\a");
  ASSERT_NORMALIZE("c:/a/..", "c:\\");
  ASSERT_NORMALIZE("c:/a/../", "c:\\");
  ASSERT_NORMALIZE("c:/a/./../", "c:\\");
  ASSERT_NORMALIZE("c:/../d:/e", "c:\\d:\\e");
  ASSERT_NORMALIZE("c:/../d:/../e", "c:\\e");

  ASSERT_NORMALIZE("foo", "foo");
  ASSERT_NORMALIZE("foo/", "foo");
  ASSERT_NORMALIZE("foo//bar", "foo\\bar");
  ASSERT_NORMALIZE("../..//foo/./bar", "..\\..\\foo\\bar");
  ASSERT_NORMALIZE("../foo/baz/../bar", "..\\foo\\bar");
  ASSERT_NORMALIZE("c:", "c:\\");
  ASSERT_NORMALIZE("c:/", "c:\\");
  ASSERT_NORMALIZE("c:\\", "c:\\");
  ASSERT_NORMALIZE("c:\\..//foo/./bar/", "c:\\foo\\bar");
  ASSERT_NORMALIZE("../foo", "..\\foo");
#undef ASSERT_NORMALIZE
}

TEST(FileTests, TestIsVolumeMountPointTarget) {
  // Volume mount points ("mounted folders") point at the root of a volume.
  EXPECT_TRUE(IsVolumeMountPointTarget(
      L"\\??\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\"));
  EXPECT_TRUE(IsVolumeMountPointTarget(
      L"\\??\\VOLUME{4E768E80-BF1F-11F1-8828-00155D01CC0A}\\"));

  // Junctions point at a directory.
  EXPECT_FALSE(IsVolumeMountPointTarget(L"\\??\\C:\\"));
  EXPECT_FALSE(IsVolumeMountPointTarget(L"\\??\\C:\\dir"));
  EXPECT_FALSE(IsVolumeMountPointTarget(L"\\??\\UNC\\server\\share\\dir"));
  EXPECT_FALSE(IsVolumeMountPointTarget(
      L"\\??\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\dir"));
  EXPECT_FALSE(IsVolumeMountPointTarget(
      L"\\??\\C:\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\"));

  // Malformed names.
  EXPECT_FALSE(IsVolumeMountPointTarget(L""));
  EXPECT_FALSE(IsVolumeMountPointTarget(L"\\??\\Volume{"));
  EXPECT_FALSE(IsVolumeMountPointTarget(
      L"\\??\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}"));
  EXPECT_FALSE(IsVolumeMountPointTarget(
      L"Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\"));
  EXPECT_FALSE(IsVolumeMountPointTarget(
      L"\\\\?\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\"));
}

// Builds reparse data as FSCTL_GET_REPARSE_POINT returns it for a symlink
// (IO_REPARSE_TAG_SYMLINK) or a junction or volume mount point
// (IO_REPARSE_TAG_MOUNT_POINT) whose substitute name is `substitute_name`. See
// https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_reparse_data_buffer
static std::vector<uint8_t> MakeReparseData(ULONG tag,
                                            const wstring& substitute_name) {
  // Header: ULONG ReparseTag, USHORT ReparseDataLength, USHORT Reserved.
  // Then USHORT SubstituteNameOffset, SubstituteNameLength, PrintNameOffset,
  // PrintNameLength; then ULONG Flags (symlinks only); then PathBuffer, which
  // holds the null-terminated substitute name and an empty print name.
  const size_t path_buffer_offset = tag == IO_REPARSE_TAG_SYMLINK ? 20 : 16;
  const USHORT name_length =
      static_cast<USHORT>(substitute_name.size() * sizeof(WCHAR));
  std::vector<uint8_t> data(path_buffer_offset + name_length +
                            2 * sizeof(WCHAR));
  const USHORT data_length = static_cast<USHORT>(data.size() - 8);
  const USHORT names[] = {/* SubstituteNameOffset */ 0, name_length,
                          /* PrintNameOffset */
                          static_cast<USHORT>(name_length + sizeof(WCHAR)),
                          /* PrintNameLength */ 0};
  memcpy(data.data(), &tag, sizeof(tag));
  memcpy(data.data() + 4, &data_length, sizeof(data_length));
  memcpy(data.data() + 8, names, sizeof(names));
  memcpy(data.data() + path_buffer_offset, substitute_name.c_str(),
         name_length);
  return data;
}

static int Interpret(const std::vector<uint8_t>& data, wstring* target) {
  return InterpretReparseData(data.data(), data.size(), target);
}

// Regression test for volume mount points ("mounted folders"): they share the
// junction reparse tag, but must not be followed as links. Unlike
// TestVolumeMountPointIsNotALink this needs no privileges, so it always runs.
TEST(FileTests, TestInterpretReparseDataOfVolumeMountPoint) {
  wstring target;
  EXPECT_EQ(
      Interpret(MakeReparseData(
                    IO_REPARSE_TAG_MOUNT_POINT,
                    L"\\??\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\"),
                &target),
      ReadSymlinkOrJunctionResult::kNotALink);
  EXPECT_EQ(target, L"");
}

TEST(FileTests, TestInterpretReparseDataOfJunction) {
  wstring target;
  EXPECT_EQ(Interpret(MakeReparseData(IO_REPARSE_TAG_MOUNT_POINT,
                                      L"\\??\\C:\\some\\dir"),
                      &target),
            ReadSymlinkOrJunctionResult::kSuccess);
  EXPECT_EQ(target, L"\\??\\C:\\some\\dir");

  // A junction into a directory of a volume, addressed by its GUID path, is
  // still a link.
  const wstring into_volume =
      L"\\??\\Volume{4e768e80-bf1f-11f1-8828-00155d01cc0a}\\subdir";
  EXPECT_EQ(Interpret(MakeReparseData(IO_REPARSE_TAG_MOUNT_POINT, into_volume),
                      &target),
            ReadSymlinkOrJunctionResult::kSuccess);
  EXPECT_EQ(target, into_volume);
}

TEST(FileTests, TestInterpretReparseDataOfSymlink) {
  wstring target;
  EXPECT_EQ(Interpret(MakeReparseData(IO_REPARSE_TAG_SYMLINK,
                                      L"\\??\\C:\\some\\file"),
                      &target),
            ReadSymlinkOrJunctionResult::kSuccess);
  EXPECT_EQ(target, L"\\??\\C:\\some\\file");
}

TEST(FileTests, TestInterpretReparseDataOfOtherTags) {
  wstring target;
  EXPECT_EQ(Interpret(MakeReparseData(IO_REPARSE_TAG_PROJFS, L""), &target),
            ReadSymlinkOrJunctionResult::kNotALink);
  EXPECT_EQ(Interpret(MakeReparseData(IO_REPARSE_TAG_DEDUP, L""), &target),
            ReadSymlinkOrJunctionResult::kError);
}

TEST(FileTests, TestInterpretReparseDataRejectsMalformedData) {
  wstring target;
  EXPECT_EQ(InterpretReparseData(nullptr, 0, &target),
            ReadSymlinkOrJunctionResult::kError);

  // Every truncation that cuts into the substitute name is an error. Copy the
  // prefix into its own buffer so that reading past it is a real overread.
  const wstring name = L"\\??\\C:\\some\\dir";
  for (ULONG tag : {IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK}) {
    const std::vector<uint8_t> full = MakeReparseData(tag, name);
    const size_t name_end = full.size() - 2 * sizeof(WCHAR);
    for (size_t size = 0; size < name_end; ++size) {
      std::vector<uint8_t> truncated(full.begin(), full.begin() + size);
      EXPECT_EQ(Interpret(truncated, &target),
                ReadSymlinkOrJunctionResult::kError)
          << "tag=" << tag << " size=" << size;
    }
  }

  // A substitute name offset pointing past the end of the data.
  std::vector<uint8_t> bad_offset =
      MakeReparseData(IO_REPARSE_TAG_MOUNT_POINT, name);
  const USHORT kHugeOffset = 0xFFF0;
  memcpy(bad_offset.data() + 8, &kHugeOffset, sizeof(kHugeOffset));
  EXPECT_EQ(Interpret(bad_offset, &target),
            ReadSymlinkOrJunctionResult::kError);
}

}  // namespace windows
}  // namespace bazel
