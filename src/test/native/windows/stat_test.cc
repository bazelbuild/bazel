// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include "src/main/native/windows/stat.h"

#include <windows.h>

#include <cstdint>
#include <string>

#include "gtest/gtest.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"
#include "src/test/cpp/util/windows_test_util.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace bazel {
namespace windows {

using blaze_util::CreateDummyFile;
using blaze_util::DeleteAllUnder;
using blaze_util::GetTestTmpDirW;
using std::string;

static const std::wstring kUncPrefix = std::wstring(L"\\\\?\\");
static const string kContent = "hello";

static bool HasGetFileInformationByName() {
  return ::GetProcAddress(::GetModuleHandleW(L"kernel32.dll"),
                          "GetFileInformationByName") != nullptr;
}

static int64_t ToTicks(const FILETIME& time) {
  return static_cast<int64_t>(
      (static_cast<uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime);
}

class WindowsStatTest : public ::testing::Test {
 public:
  void TearDown() override { DeleteAllUnder(GetTestTmpDirW()); }

 protected:
  static std::wstring TmpDir() { return kUncPrefix + GetTestTmpDirW(); }
};

TEST_F(WindowsStatTest, TestReportsPathThatIsNotAbsoluteAndNormalized) {
  FileMetadata metadata;
  for (const wchar_t* path : {L"", L"foo\\bar", L"c:\\foo\\..\\bar"}) {
    std::wstring error;
    EXPECT_EQ(StatResult::kError, Stat(path, false, metadata, error))
        << "path=" << path;
    EXPECT_FALSE(error.empty()) << "path=" << path;
  }
}

TEST_F(WindowsStatTest, TestReportsErrorForInvalidName) {
  std::wstring path(TmpDir() + L"\\in<valid");
  FileMetadata metadata;
  std::wstring error;
  EXPECT_EQ(StatResult::kError, Stat(path, false, metadata, error));
  EXPECT_FALSE(error.empty());
}

TEST_F(WindowsStatTest, TestReportsMissingPath) {
  std::wstring path(TmpDir() + L"\\no_such_entry");
  FileMetadata metadata;
  std::wstring error;
  EXPECT_EQ(StatResult::kDoesNotExist,
            Stat(path.c_str(), false, metadata, error));
  EXPECT_EQ(StatResult::kDoesNotExist,
            Stat(path.c_str(), true, metadata, error));
}

TEST_F(WindowsStatTest, TestDescribesRegularFile) {
  std::wstring path(TmpDir() + L"\\a_file");
  ASSERT_TRUE(CreateDummyFile(path, kContent));

  for (bool follow : {false, true}) {
    FileMetadata metadata;
    std::wstring error;
    ASSERT_EQ(StatResult::kSuccess,
              Stat(path.c_str(), follow, metadata, error))
        << "follow=" << follow;
    EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);
    EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
    EXPECT_EQ(static_cast<int64_t>(kContent.size()), metadata.size);
    EXPECT_GT(metadata.last_write_time.QuadPart, 0);
    EXPECT_GT(metadata.change_time.QuadPart, 0);
  }
}

TEST_F(WindowsStatTest, TestDescribesDirectory) {
  std::wstring path(TmpDir() + L"\\a_dir");
  ASSERT_TRUE(::CreateDirectoryW(path.c_str(), nullptr));

  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(path.c_str(), false, metadata, error));
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);
  EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
}

TEST_F(WindowsStatTest, TestDescribesRootDirectory) {
  std::wstring root(TmpDir().substr(0, kUncPrefix.size() + 3));
  ASSERT_EQ(L'\\', root[root.size() - 1]);

  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(root.c_str(), false, metadata, error));
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);
}

TEST_F(WindowsStatTest, TestDescribesJunctionAndFollowsIt) {
  std::wstring target(TmpDir() + L"\\junc_target");
  std::wstring name(TmpDir() + L"\\junc");
  ASSERT_TRUE(::CreateDirectoryW(target.c_str(), nullptr));
  ASSERT_EQ(CreateJunctionResult::kSuccess,
            CreateJunction(name, target, nullptr));

  // NOFOLLOW describes the junction itself.
  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(name.c_str(), false, metadata, error));
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);

  // FOLLOW describes the target.
  ASSERT_EQ(StatResult::kSuccess, Stat(name.c_str(), true, metadata, error));
  EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);
}

TEST_F(WindowsStatTest, TestDescribesDanglingJunctionUnderNofollow) {
  std::wstring target(TmpDir() + L"\\gone");
  std::wstring name(TmpDir() + L"\\dangling_junc");
  ASSERT_TRUE(::CreateDirectoryW(target.c_str(), nullptr));
  ASSERT_EQ(CreateJunctionResult::kSuccess,
            CreateJunction(name, target, nullptr));
  ASSERT_TRUE(::RemoveDirectoryW(target.c_str()));

  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(name.c_str(), false, metadata, error));
  EXPECT_NE(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
}

TEST_F(WindowsStatTest, TestDescribesFileReachedThroughJunction) {
  // Junctions in the middle of the path are always followed.
  std::wstring target(TmpDir() + L"\\via_target");
  std::wstring name(TmpDir() + L"\\via_junc");
  ASSERT_TRUE(::CreateDirectoryW(target.c_str(), nullptr));
  ASSERT_TRUE(CreateDummyFile(target + L"\\a_file", kContent));
  ASSERT_EQ(CreateJunctionResult::kSuccess,
            CreateJunction(name, target, nullptr));

  std::wstring path(name + L"\\a_file");
  for (bool follow : {false, true}) {
    FileMetadata metadata;
    std::wstring error;
    ASSERT_EQ(StatResult::kSuccess,
              Stat(path.c_str(), follow, metadata, error))
        << "follow=" << follow;
    EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_REPARSE_POINT);
    EXPECT_EQ(0u, metadata.attributes & FILE_ATTRIBUTE_DIRECTORY);
    EXPECT_EQ(static_cast<int64_t>(kContent.size()), metadata.size);
  }
}

TEST_F(WindowsStatTest, TestDescribesFileAnotherHandleHoldsExclusively) {
  // GetFileInformationByName opens no handle, so sharing mode doesn't matter.
  if (!HasGetFileInformationByName()) {
    GTEST_SKIP() << "this machine has no GetFileInformationByName";
  }
  std::wstring path(TmpDir() + L"\\exclusive");
  ASSERT_TRUE(CreateDummyFile(path, kContent));

  AutoHandle handle(::CreateFileW(path.c_str(), GENERIC_READ,
                                  /* dwShareMode */ 0, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr));
  ASSERT_TRUE(handle.IsValid());

  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(path.c_str(), false, metadata, error));
  EXPECT_EQ(static_cast<int64_t>(kContent.size()), metadata.size);
}

TEST_F(WindowsStatTest, TestReadsTheSameFieldsAsTheCallsItReplaces) {
  std::wstring path(TmpDir() + L"\\a_file");
  ASSERT_TRUE(CreateDummyFile(path, kContent));

  FileMetadata metadata;
  std::wstring error;
  ASSERT_EQ(StatResult::kSuccess, Stat(path.c_str(), false, metadata, error));

  WIN32_FILE_ATTRIBUTE_DATA expected = {};
  ASSERT_TRUE(
      ::GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &expected));
  int64_t expected_change_time = 0;
  ASSERT_EQ(GetChangeTimeResult::kSuccess,
            GetChangeTime(path.c_str(), false, &expected_change_time, nullptr));

  // Checks that the local FILE_STAT_BASIC_INFORMATION layout matches the SDK's.
  EXPECT_EQ(expected.dwFileAttributes, metadata.attributes);
  EXPECT_EQ((static_cast<int64_t>(expected.nFileSizeHigh) << 32) |
                expected.nFileSizeLow,
            metadata.size);
  EXPECT_EQ(ToTicks(expected.ftLastWriteTime),
            metadata.last_write_time.QuadPart);
  EXPECT_EQ(expected_change_time,
            WindowsFileTimeToUnixNanos(metadata.change_time));
}

}  // namespace windows
}  // namespace bazel
