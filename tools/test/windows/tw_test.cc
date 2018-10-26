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
#include <vector>

#include "gtest/gtest.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/native/windows/file.h"
#include "src/test/cpp/util/windows_test_util.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/zip.h"
#include "tools/test/windows/tw.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace {

using bazel::tools::test_wrapper::FileInfo;
using bazel::tools::test_wrapper::ZipEntryPaths;
using bazel::tools::test_wrapper::testing::TestOnly_AsMixedPath;
using bazel::tools::test_wrapper::testing::TestOnly_CreateZip;
using bazel::tools::test_wrapper::testing::TestOnly_GetEnv;
using bazel::tools::test_wrapper::testing::TestOnly_GetFileListRelativeTo;
using bazel::tools::test_wrapper::testing::TestOnly_ToZipEntryPaths;

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

// According to this StackOverflow post [1] `const` modifies what's on its
// *left*, and "const char" is equivalent to "char const".
// `ZipBuilder::EstimateSize`'s argument type is "char const* const*" meaning a
// mutable array (right *) of const pointers (left *) to const data (char).
//
// [1] https://stackoverflow.com/a/8091846/7778502
void CompareZipEntryPaths(char const* const* actual,
                          std::vector<const char*> expected, int line) {
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NE(actual[i], nullptr)
        << __FILE__ << "(" << line << "): assertion failed here";
    EXPECT_EQ(std::string(actual[i]), std::string(expected[i]))
        << __FILE__ << "(" << line << "): assertion failed here";
  }
  EXPECT_EQ(actual[expected.size()], nullptr)
      << __FILE__ << "(" << line << "): assertion failed here; value was ("
      << actual[expected.size()] << ")";
}

#define GET_TEST_TMPDIR(result) GetTestTmpdir(result, __LINE__)
#define CREATE_JUNCTION(name, target) CreateJunction(name, target, __LINE__)
#define COMPARE_FILE_INFOS(actual, expected) \
  CompareFileInfos(actual, expected, __LINE__)
#define COMPARE_ZIP_ENTRY_PATHS(actual, expected) \
  CompareZipEntryPaths(actual, expected, __LINE__)

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

TEST_F(TestWrapperWindowsTest, TestToZipEntryPaths) {
  // Pretend we already acquired a file list. The files don't have to exist.
  std::wstring root = L"c:\\nul\\root";
  std::vector<FileInfo> files = {
      {L"foo\\sub\\file1", 0},  {L"foo\\sub\\file2", 5},
      {L"foo\\file1", 3},       {L"foo\\file2", 6},
      {L"foo\\junc\\file1", 0}, {L"foo\\junc\\file2", 5}};

  ZipEntryPaths actual;
  ASSERT_TRUE(TestOnly_ToZipEntryPaths(root, files, &actual));
  ASSERT_EQ(actual.Size(), 6);

  std::vector<const char*> expected_abs_paths = {
      "c:/nul/root/foo/sub/file1",  "c:/nul/root/foo/sub/file2",
      "c:/nul/root/foo/file1",      "c:/nul/root/foo/file2",
      "c:/nul/root/foo/junc/file1", "c:/nul/root/foo/junc/file2",
  };
  COMPARE_ZIP_ENTRY_PATHS(actual.AbsPathPtrs(), expected_abs_paths);

  std::vector<const char*> expected_entry_paths = {
      "foo/sub/file1", "foo/sub/file2",  "foo/file1",
      "foo/file2",     "foo/junc/file1", "foo/junc/file2",
  };
  COMPARE_ZIP_ENTRY_PATHS(actual.EntryPathPtrs(), expected_entry_paths);
}

TEST_F(TestWrapperWindowsTest, TestToZipEntryPathsLongPathRoot) {
  // Pretend we already acquired a file list. The files don't have to exist.
  // Assert that the root is allowed to have the `\\?\` prefix, but the zip
  // entry paths won't have it.
  std::wstring root = L"\\\\?\\c:\\nul\\unc";
  std::vector<FileInfo> files = {
      {L"foo\\sub\\file1", 0},  {L"foo\\sub\\file2", 5},
      {L"foo\\file1", 3},       {L"foo\\file2", 6},
      {L"foo\\junc\\file1", 0}, {L"foo\\junc\\file2", 5}};

  ZipEntryPaths actual;
  ASSERT_TRUE(TestOnly_ToZipEntryPaths(root, files, &actual));
  ASSERT_EQ(actual.Size(), 6);

  std::vector<const char*> expected_abs_paths = {
      "c:/nul/unc/foo/sub/file1",  "c:/nul/unc/foo/sub/file2",
      "c:/nul/unc/foo/file1",      "c:/nul/unc/foo/file2",
      "c:/nul/unc/foo/junc/file1", "c:/nul/unc/foo/junc/file2",
  };
  COMPARE_ZIP_ENTRY_PATHS(actual.AbsPathPtrs(), expected_abs_paths);

  std::vector<const char*> expected_entry_paths = {
      "foo/sub/file1", "foo/sub/file2",  "foo/file1",
      "foo/file2",     "foo/junc/file1", "foo/junc/file2",
  };
  COMPARE_ZIP_ENTRY_PATHS(actual.EntryPathPtrs(), expected_entry_paths);
}

class InMemoryExtractor : public devtools_ijar::ZipExtractorProcessor {
 public:
  struct ExtractedFile {
    std::string path;
    std::unique_ptr<devtools_ijar::u1[]> data;
    size_t size;
  };

  InMemoryExtractor(std::vector<ExtractedFile>* extracted)
      : extracted_(extracted) {}

  bool Accept(const char* filename, const devtools_ijar::u4 attr) override {
    return true;
  }

  void Process(const char* filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1* data, const size_t size) override {
    extracted_->push_back({});
    extracted_->back().path = filename;
    extracted_->back().data.reset(new devtools_ijar::u1[size]);
    memcpy(extracted_->back().data.get(), data, size);
    extracted_->back().size = size;
  }

 private:
  std::vector<ExtractedFile>* extracted_;
};

TEST_F(TestWrapperWindowsTest, TestCreateZip) {
  std::wstring tmpdir;
  GET_TEST_TMPDIR(&tmpdir);

  // Create a directory structure to archive.
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

  std::vector<FileInfo> file_list = {
      {L"foo\\sub\\file1", 0},  {L"foo\\sub\\file2", 5},
      {L"foo\\file1", 3},       {L"foo\\file2", 6},
      {L"foo\\junc\\file1", 0}, {L"foo\\junc\\file2", 5}};

  ASSERT_TRUE(TestOnly_CreateZip(root, file_list, root + L"\\x.zip"));

  std::string zip_path;
  EXPECT_TRUE(TestOnly_AsMixedPath(root + L"\\x.zip", &zip_path));

  // Extract the zip file into memory to verify its contents.
  std::vector<InMemoryExtractor::ExtractedFile> extracted;
  InMemoryExtractor extractor(&extracted);
  std::unique_ptr<devtools_ijar::ZipExtractor> zip(
      devtools_ijar::ZipExtractor::Create(zip_path.c_str(), &extractor));
  EXPECT_NE(zip.get(), nullptr);
  EXPECT_EQ(zip->ProcessAll(), 0);

  EXPECT_EQ(extracted.size(), 6);

  EXPECT_EQ(extracted[0].path, std::string("foo/sub/file1"));
  EXPECT_EQ(extracted[1].path, std::string("foo/sub/file2"));
  EXPECT_EQ(extracted[2].path, std::string("foo/file1"));
  EXPECT_EQ(extracted[3].path, std::string("foo/file2"));
  EXPECT_EQ(extracted[4].path, std::string("foo/junc/file1"));
  EXPECT_EQ(extracted[5].path, std::string("foo/junc/file2"));

  EXPECT_EQ(extracted[0].size, 0);
  EXPECT_EQ(extracted[1].size, 5);
  EXPECT_EQ(extracted[2].size, 3);
  EXPECT_EQ(extracted[3].size, 6);
  EXPECT_EQ(extracted[4].size, 0);
  EXPECT_EQ(extracted[5].size, 5);

  EXPECT_EQ(memcmp(extracted[1].data.get(), "hello", 5), 0);
  EXPECT_EQ(memcmp(extracted[2].data.get(), "foo", 3), 0);
  EXPECT_EQ(memcmp(extracted[3].data.get(), "foobar", 6), 0);
  EXPECT_EQ(memcmp(extracted[5].data.get(), "hello", 5), 0);
}

}  // namespace
