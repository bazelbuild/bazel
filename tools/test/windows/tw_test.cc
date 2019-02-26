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
#include "src/main/native/windows/util.h"
#include "src/test/cpp/util/windows_test_util.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/zip.h"
#include "tools/test/windows/tw.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace {

using bazel::tools::test_wrapper::FileInfo;
using bazel::tools::test_wrapper::IFStream;
using bazel::tools::test_wrapper::ZipEntryPaths;
using bazel::tools::test_wrapper::testing::TestOnly_AsMixedPath;
using bazel::tools::test_wrapper::testing::TestOnly_CdataEncode;
using bazel::tools::test_wrapper::testing::TestOnly_CreateIFStream;
using bazel::tools::test_wrapper::testing::TestOnly_CreateTee;
using bazel::tools::test_wrapper::testing::
    TestOnly_CreateUndeclaredOutputsAnnotations;
using bazel::tools::test_wrapper::testing::
    TestOnly_CreateUndeclaredOutputsManifest;
using bazel::tools::test_wrapper::testing::TestOnly_CreateZip;
using bazel::tools::test_wrapper::testing::TestOnly_GetEnv;
using bazel::tools::test_wrapper::testing::TestOnly_GetFileListRelativeTo;
using bazel::tools::test_wrapper::testing::TestOnly_GetMimeType;
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
              return a.RelativePath() > b.RelativePath();
            });
  std::sort(expected.begin(), expected.end(),
            [](const FileInfo& a, const FileInfo& b) {
              return a.RelativePath() > b.RelativePath();
            });
  for (std::vector<FileInfo>::size_type i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].RelativePath(), expected[i].RelativePath())
        << __FILE__ << "(" << line << "): assertion failed here; index: " << i;
    ASSERT_EQ(actual[i].Size(), expected[i].Size())
        << __FILE__ << "(" << line << "): assertion failed here; index: " << i;
    ASSERT_EQ(actual[i].IsDirectory(), expected[i].IsDirectory())
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

HANDLE FopenRead(const std::wstring& unc_path) {
  return CreateFileW(unc_path.c_str(), GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                     FILE_ATTRIBUTE_NORMAL, NULL);
}

HANDLE FopenContents(wchar_t* wline, const char* contents) {
  std::wstring tmpdir;
  GET_TEST_TMPDIR(&tmpdir);
  std::wstring filename = tmpdir + L"\\tmp" + wline;
  EXPECT_TRUE(blaze_util::CreateDummyFile(filename, contents));
  return FopenRead(filename);
}

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

  std::vector<FileInfo> expected = {FileInfo(L"foo"),
                                    FileInfo(L"foo\\sub"),
                                    FileInfo(L"foo\\sub\\file1", 0),
                                    FileInfo(L"foo\\sub\\file2", 5),
                                    FileInfo(L"foo\\file1", 3),
                                    FileInfo(L"foo\\file2", 6),
                                    FileInfo(L"foo\\junc"),
                                    FileInfo(L"foo\\junc\\file1", 0),
                                    FileInfo(L"foo\\junc\\file2", 5)};
  COMPARE_FILE_INFOS(actual, expected);

  // Assert traversal of "foo" -- should include all files, but now with paths
  // relative to "foo".
  actual.clear();
  ASSERT_TRUE(
      TestOnly_GetFileListRelativeTo((root + L"\\foo").c_str(), &actual));

  expected = {FileInfo(L"sub"),
              FileInfo(L"sub\\file1", 0),
              FileInfo(L"sub\\file2", 5),
              FileInfo(L"file1", 3),
              FileInfo(L"file2", 6),
              FileInfo(L"junc"),
              FileInfo(L"junc\\file1", 0),
              FileInfo(L"junc\\file2", 5)};
  COMPARE_FILE_INFOS(actual, expected);

  // Assert traversal limited to the current directory (depth of 0).
  actual.clear();
  ASSERT_TRUE(TestOnly_GetFileListRelativeTo(root, &actual, 0));
  expected = {FileInfo(L"foo")};
  COMPARE_FILE_INFOS(actual, expected);

  // Assert traversal limited to depth of 1.
  actual.clear();
  ASSERT_TRUE(TestOnly_GetFileListRelativeTo(root, &actual, 1));
  expected = {FileInfo(L"foo"), FileInfo(L"foo\\sub"),
              FileInfo(L"foo\\file1", 3), FileInfo(L"foo\\file2", 6),
              FileInfo(L"foo\\junc")};
  COMPARE_FILE_INFOS(actual, expected);
}

TEST_F(TestWrapperWindowsTest, TestToZipEntryPaths) {
  // Pretend we already acquired a file list. The files don't have to exist.
  std::wstring root = L"c:\\nul\\root";
  std::vector<FileInfo> files = {FileInfo(L"foo"),
                                 FileInfo(L"foo\\sub"),
                                 FileInfo(L"foo\\sub\\file1", 0),
                                 FileInfo(L"foo\\sub\\file2", 5),
                                 FileInfo(L"foo\\file1", 3),
                                 FileInfo(L"foo\\file2", 6),
                                 FileInfo(L"foo\\junc"),
                                 FileInfo(L"foo\\junc\\file1", 0),
                                 FileInfo(L"foo\\junc\\file2", 5)};

  ZipEntryPaths actual;
  ASSERT_TRUE(TestOnly_ToZipEntryPaths(root, files, &actual));
  ASSERT_EQ(actual.Size(), 9);

  std::vector<const char*> expected_abs_paths = {
      "c:/nul/root/foo/",          "c:/nul/root/foo/sub/",
      "c:/nul/root/foo/sub/file1", "c:/nul/root/foo/sub/file2",
      "c:/nul/root/foo/file1",     "c:/nul/root/foo/file2",
      "c:/nul/root/foo/junc/",     "c:/nul/root/foo/junc/file1",
      "c:/nul/root/foo/junc/file2"};
  COMPARE_ZIP_ENTRY_PATHS(actual.AbsPathPtrs(), expected_abs_paths);

  std::vector<const char*> expected_entry_paths = {
      "foo/",      "foo/sub/",  "foo/sub/file1",  "foo/sub/file2", "foo/file1",
      "foo/file2", "foo/junc/", "foo/junc/file1", "foo/junc/file2"};
  COMPARE_ZIP_ENTRY_PATHS(actual.EntryPathPtrs(), expected_entry_paths);
}

TEST_F(TestWrapperWindowsTest, TestToZipEntryPathsLongPathRoot) {
  // Pretend we already acquired a file list. The files don't have to exist.
  // Assert that the root is allowed to have the `\\?\` prefix, but the zip
  // entry paths won't have it.
  std::wstring root = L"\\\\?\\c:\\nul\\unc";
  std::vector<FileInfo> files = {FileInfo(L"foo"),
                                 FileInfo(L"foo\\sub"),
                                 FileInfo(L"foo\\sub\\file1", 0),
                                 FileInfo(L"foo\\sub\\file2", 5),
                                 FileInfo(L"foo\\file1", 3),
                                 FileInfo(L"foo\\file2", 6),
                                 FileInfo(L"foo\\junc"),
                                 FileInfo(L"foo\\junc\\file1", 0),
                                 FileInfo(L"foo\\junc\\file2", 5)};

  ZipEntryPaths actual;
  ASSERT_TRUE(TestOnly_ToZipEntryPaths(root, files, &actual));
  ASSERT_EQ(actual.Size(), 9);

  std::vector<const char*> expected_abs_paths = {
      "c:/nul/unc/foo/",          "c:/nul/unc/foo/sub/",
      "c:/nul/unc/foo/sub/file1", "c:/nul/unc/foo/sub/file2",
      "c:/nul/unc/foo/file1",     "c:/nul/unc/foo/file2",
      "c:/nul/unc/foo/junc/",     "c:/nul/unc/foo/junc/file1",
      "c:/nul/unc/foo/junc/file2"};
  COMPARE_ZIP_ENTRY_PATHS(actual.AbsPathPtrs(), expected_abs_paths);

  std::vector<const char*> expected_entry_paths = {
      "foo/",      "foo/sub/",  "foo/sub/file1",  "foo/sub/file2", "foo/file1",
      "foo/file2", "foo/junc/", "foo/junc/file1", "foo/junc/file2"};
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
    if (size > 0) {
      extracted_->back().data.reset(new devtools_ijar::u1[size]);
      memcpy(extracted_->back().data.get(), data, size);
    }
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

  std::vector<FileInfo> file_list = {FileInfo(L"foo"),
                                     FileInfo(L"foo\\sub"),
                                     FileInfo(L"foo\\sub\\file1", 0),
                                     FileInfo(L"foo\\sub\\file2", 5),
                                     FileInfo(L"foo\\file1", 3),
                                     FileInfo(L"foo\\file2", 6),
                                     FileInfo(L"foo\\junc"),
                                     FileInfo(L"foo\\junc\\file1", 0),
                                     FileInfo(L"foo\\junc\\file2", 5)};

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

  EXPECT_EQ(extracted.size(), 9);

  EXPECT_EQ(extracted[0].path, std::string("foo/"));
  EXPECT_EQ(extracted[1].path, std::string("foo/sub/"));
  EXPECT_EQ(extracted[2].path, std::string("foo/sub/file1"));
  EXPECT_EQ(extracted[3].path, std::string("foo/sub/file2"));
  EXPECT_EQ(extracted[4].path, std::string("foo/file1"));
  EXPECT_EQ(extracted[5].path, std::string("foo/file2"));
  EXPECT_EQ(extracted[6].path, std::string("foo/junc/"));
  EXPECT_EQ(extracted[7].path, std::string("foo/junc/file1"));
  EXPECT_EQ(extracted[8].path, std::string("foo/junc/file2"));

  EXPECT_EQ(extracted[0].size, 0);
  EXPECT_EQ(extracted[1].size, 0);
  EXPECT_EQ(extracted[2].size, 0);
  EXPECT_EQ(extracted[3].size, 5);
  EXPECT_EQ(extracted[4].size, 3);
  EXPECT_EQ(extracted[5].size, 6);
  EXPECT_EQ(extracted[6].size, 0);
  EXPECT_EQ(extracted[7].size, 0);
  EXPECT_EQ(extracted[8].size, 5);

  EXPECT_EQ(memcmp(extracted[3].data.get(), "hello", 5), 0);
  EXPECT_EQ(memcmp(extracted[4].data.get(), "foo", 3), 0);
  EXPECT_EQ(memcmp(extracted[5].data.get(), "foobar", 6), 0);
  EXPECT_EQ(memcmp(extracted[8].data.get(), "hello", 5), 0);
}

TEST_F(TestWrapperWindowsTest, TestGetMimeType) {
  // As of 2018-11-08, TestOnly_GetMimeType looks up the MIME type from the
  // registry under `HKCR\<extension>\Content Type`, e.g.
  // 'HKCR\.bmp\Content Type`.
  // Bazel's CI machines run Windows Server 2016 Core, whose registry contains
  // the Content Type for .ico and .bmp but not for common types such as .txt,
  // hence the file types we choose to test for.
  EXPECT_EQ(TestOnly_GetMimeType("foo.ico"), std::string("image/x-icon"));
  EXPECT_EQ(TestOnly_GetMimeType("foo.bmp"), std::string("image/bmp"));
  EXPECT_EQ(TestOnly_GetMimeType("foo"),
            std::string("application/octet-stream"));
}

TEST_F(TestWrapperWindowsTest, TestUndeclaredOutputsManifest) {
  // Pretend we already acquired a file list. The files don't have to exist.
  // Assert that the root is allowed to have the `\\?\` prefix, but the zip
  // entry paths won't have it.
  std::vector<FileInfo> files = {FileInfo(L"foo"),
                                 FileInfo(L"foo\\sub"),
                                 FileInfo(L"foo\\sub\\file1.ico", 0),
                                 FileInfo(L"foo\\sub\\file2.bmp", 5),
                                 FileInfo(L"foo\\file2", 6)};

  std::string content;
  ASSERT_TRUE(TestOnly_CreateUndeclaredOutputsManifest(files, &content));
  ASSERT_EQ(content, std::string("foo/sub/file1.ico\t0\timage/x-icon\n"
                                 "foo/sub/file2.bmp\t5\timage/bmp\n"
                                 "foo/file2\t6\tapplication/octet-stream\n"));
}

TEST_F(TestWrapperWindowsTest, TestCreateUndeclaredOutputsAnnotations) {
  std::wstring tmpdir;
  GET_TEST_TMPDIR(&tmpdir);

  // Create a directory structure to parse.
  std::wstring root = tmpdir + L"\\tmp" + WLINE;
  EXPECT_TRUE(CreateDirectoryW(root.c_str(), NULL));
  EXPECT_TRUE(CreateDirectoryW((root + L"\\foo").c_str(), NULL));
  EXPECT_TRUE(CreateDirectoryW((root + L"\\bar.part").c_str(), NULL));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\a.part", "Hello a"));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\b.txt", "Hello b"));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\c.part", "Hello c"));
  EXPECT_TRUE(blaze_util::CreateDummyFile(root + L"\\foo\\d.part", "Hello d"));
  EXPECT_TRUE(
      blaze_util::CreateDummyFile(root + L"\\bar.part\\e.part", "Hello e"));

  std::wstring annot = root + L"\\x.annot";
  ASSERT_TRUE(TestOnly_CreateUndeclaredOutputsAnnotations(root, annot));

  HANDLE h = FopenRead(annot);
  ASSERT_NE(h, INVALID_HANDLE_VALUE);
  char content[100];
  DWORD read;
  bool success = ReadFile(h, content, 100, &read, NULL) != FALSE;
  CloseHandle(h);
  EXPECT_TRUE(success);
  ASSERT_EQ(std::string(content, read), std::string("Hello aHello c"));
}

TEST_F(TestWrapperWindowsTest, TestTee) {
  HANDLE read1_h, write1_h;
  EXPECT_TRUE(CreatePipe(&read1_h, &write1_h, NULL, 0));
  bazel::windows::AutoHandle read1(read1_h), write1(write1_h);
  HANDLE read2_h, write2_h;
  EXPECT_TRUE(CreatePipe(&read2_h, &write2_h, NULL, 0));
  bazel::windows::AutoHandle read2(read2_h), write2(write2_h);
  HANDLE read3_h, write3_h;
  EXPECT_TRUE(CreatePipe(&read3_h, &write3_h, NULL, 0));
  bazel::windows::AutoHandle read3(read3_h), write3(write3_h);

  std::unique_ptr<bazel::tools::test_wrapper::Tee> tee;
  EXPECT_TRUE(TestOnly_CreateTee(&read1, &write2, &write3, &tee));

  DWORD written, read;
  char content[100];

  EXPECT_TRUE(WriteFile(write1, "hello", 5, &written, NULL));
  EXPECT_EQ(written, 5);
  EXPECT_TRUE(ReadFile(read2, content, 100, &read, NULL));
  EXPECT_EQ(read, 5);
  EXPECT_EQ(std::string(content, read), "hello");
  EXPECT_TRUE(ReadFile(read3, content, 100, &read, NULL));
  EXPECT_EQ(read, 5);
  EXPECT_EQ(std::string(content, read), "hello");

  EXPECT_TRUE(WriteFile(write1, "foo", 3, &written, NULL));
  EXPECT_EQ(written, 3);
  EXPECT_TRUE(ReadFile(read2, content, 100, &read, NULL));
  EXPECT_EQ(read, 3);
  EXPECT_EQ(std::string(content, read), "foo");
  EXPECT_TRUE(ReadFile(read3, content, 100, &read, NULL));
  EXPECT_EQ(read, 3);
  EXPECT_EQ(std::string(content, read), "foo");

  write1 = INVALID_HANDLE_VALUE;  // closes handle so the Tee thread can exit
}

void AssertCdataEncodeBuffer(const char* input, DWORD size,
                             const char* expected_output) {
  std::stringstream out_stm;
  ASSERT_TRUE(TestOnly_CdataEncode(reinterpret_cast<const uint8_t*>(input),
                                   size, &out_stm));
  ASSERT_EQ(expected_output, out_stm.str());
}

void AssertCdataEncodeBuffer(const char* input, const char* expected_output) {
  AssertCdataEncodeBuffer(input, strlen(input), expected_output);
}

TEST_F(TestWrapperWindowsTest, TestCdataEscapeNullTerminator) {
  AssertCdataEncodeBuffer("x\0y", 3, "x?y");
}

TEST_F(TestWrapperWindowsTest, TestCdataEscapeCdataEndings) {
  AssertCdataEncodeBuffer(
      // === Input ===
      // CDATA end sequence, followed by some arbitrary octet.
      "]]>x"
      // CDATA end sequence twice.
      "]]>]]>x"
      // CDATA end sequence at the end of the string.
      "]]>",

      // === Expected output ===
      "]]>]]<![CDATA[>x"
      "]]>]]<![CDATA[>]]>]]<![CDATA[>x"
      "]]>]]<![CDATA[>");
}

TEST_F(TestWrapperWindowsTest, TestCdataEscapeSingleOctets) {
  AssertCdataEncodeBuffer(  // === Input ===
                            // Legal single-octets.
      "AB\x9\xA\xD\x20\x7F"
      // Illegal single-octets.
      "\x8\xB\xC\x1F\x80\xFF"
      "x",

      // === Expected output ===
      // Legal single-octets.
      "AB\x9\xA\xD\x20\x7F"
      // Illegal single-octets.
      "??????"
      "x");
}

TEST_F(TestWrapperWindowsTest, TestCdataEscapeDoubleOctets) {
  // Legal range: [\xc0-\xdf][\x80-\xbf]
  AssertCdataEncodeBuffer(
      "x"
      // Legal double-octet sequences.
      "\xC0\x80"
      "\xDE\xB0"
      "\xDF\xBF"
      // Illegal double-octet sequences, first octet is bad, second is good.
      "\xBF\x80"  // each are matched as single bad octets
      "\xE0\x80"
      // Illegal double-octet sequences, first octet is good, second is bad.
      "\xC0\x7F"  // 0x7F is legal as a single-octet, retained
      "\xDF\xC0"  // 0xC0 starts a legal two-octet sequence...
      // Illegal double-octet sequences, both octets bad.
      "\xBF\xFF"  // ...and 0xBF finishes that sequence
      "x",

      // === Expected output ===
      "x"
      // Legal double-octet sequences.
      "\xC0\x80"
      "\xDE\xB0"
      "\xDF\xBF"
      // Illegal double-octet sequences, first octet is bad, second is good.
      "??"
      "??"
      // Illegal double-octet sequences, first octet is good, second is bad.
      "?\x7F"  // 0x7F is legal as a single-octet, retained
      "?\xC0"  // 0xC0 starts a legal two-octet sequence...
      // Illegal double-octet sequences, both octets bad.
      "\xBF?"  // ...and 0xBF finishes that sequence
      "x");
}

TEST_F(TestWrapperWindowsTest, TestCdataEscapeAndAppend) {
  std::wstring tmpdir;
  GET_TEST_TMPDIR(&tmpdir);

  AssertCdataEncodeBuffer(
      // === Input ===
      "AB\xA\xC\xD"
      "]]>"
      "]]]>"
      "\xC0\x80"
      "a"
      "\xED\x9F\xBF"
      "b"
      "\xEF\xBF\xB0"
      "c"
      "\xF7\xB0\x80\x81"
      "d"
      "]]>",

      // === Output ===
      "AB\xA?\xD"
      "]]>]]<![CDATA[>"
      "]]]>]]<![CDATA[>"
      "\xC0\x80"
      "a"
      "\xED\x9F\xBF"
      "b"
      "\xEF\xBF\xB0"
      "c"
      "\xF7\xB0\x80\x81"
      "d"
      "]]>]]<![CDATA[>");
}

TEST_F(TestWrapperWindowsTest, TestIFStreamNoData) {
  bazel::windows::AutoHandle h(FopenContents(WLINE, ""));
  std::unique_ptr<IFStream> s(TestOnly_CreateIFStream(h, 6));
  uint8_t buf[3] = {0, 0, 0};

  ASSERT_EQ(s->Get(), IFStream::kIFStreamErrorEOF);
  ASSERT_EQ(s->Peek(0, buf), 0);
  ASSERT_EQ(s->Peek(1, buf), 0);
  ASSERT_EQ(s->Peek(2, buf), 0);
  ASSERT_EQ(s->Peek(100, buf), 0);
}

TEST_F(TestWrapperWindowsTest, TestIFStreamLessDataThanPageSize) {
  // The data is "abc" (3 bytes), page size is 6 bytes.
  bazel::windows::AutoHandle h(FopenContents(WLINE, "abc"));
  std::unique_ptr<IFStream> s(TestOnly_CreateIFStream(h, 6));
  uint8_t buf[3] = {0, 0, 0};

  // Read position is at "a".
  ASSERT_EQ(s->Get(), 'a');
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'b');
  ASSERT_EQ(s->Peek(2, buf), 2);
  ASSERT_EQ(buf[0], 'b');
  ASSERT_EQ(buf[1], 'c');
  ASSERT_EQ(s->Peek(100, buf), 2);
  ASSERT_EQ(buf[0], 'b');
  ASSERT_EQ(buf[1], 'c');

  // Read position is at "b".
  ASSERT_EQ(s->Get(), 'b');
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'c');
  ASSERT_EQ(s->Peek(2, buf), 1);
  ASSERT_EQ(buf[0], 'c');
  ASSERT_EQ(s->Peek(100, buf), 1);
  ASSERT_EQ(buf[0], 'c');

  // Read position is at "c".
  ASSERT_EQ(s->Get(), 'c');
  ASSERT_EQ(s->Peek(1, buf), 0);
  ASSERT_EQ(s->Peek(2, buf), 0);
  ASSERT_EQ(s->Peek(100, buf), 0);
}

TEST_F(TestWrapperWindowsTest, TestIFStreamExactlySinglePageSize) {
  // The data is "abcdef" (6 bytes), page size is 6 bytes.
  bazel::windows::AutoHandle h(FopenContents(WLINE, "abcdef"));
  std::unique_ptr<IFStream> s(TestOnly_CreateIFStream(h, 6));
  uint8_t buf[6] = {0, 0, 0};

  // Read position is at "a".
  ASSERT_EQ(s->Get(), 'a');
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'b');
  ASSERT_EQ(s->Peek(5, buf), 5);
  ASSERT_EQ(buf[0], 'b');
  ASSERT_EQ(buf[1], 'c');
  ASSERT_EQ(buf[2], 'd');
  ASSERT_EQ(buf[3], 'e');
  ASSERT_EQ(buf[4], 'f');

  ASSERT_EQ(s->Get(), 'b');
  ASSERT_EQ(s->Get(), 'c');
  ASSERT_EQ(s->Get(), 'd');
  ASSERT_EQ(s->Get(), 'e');
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'f');
  ASSERT_EQ(s->Peek(5, buf), 1);

  // Read position is at "f". No more peeking or moving.
  ASSERT_EQ(s->Get(), 'f');
  ASSERT_EQ(s->Peek(1, buf), 0);
}

TEST_F(TestWrapperWindowsTest, TestIFStreamLessDataThanDoublePageSize) {
  bazel::windows::AutoHandle h(FopenContents(WLINE, "abcdefghi"));
  std::unique_ptr<IFStream> s(TestOnly_CreateIFStream(h, 6));
  uint8_t buf[3] = {0, 0, 0};

  // Move near the page boundary.
  for (int c = s->Get(); c != 'e'; c = s->Get()) {
  }

  // Read position is at "e". Peek2 and Peek3 will need to read from next page.
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'f');
  ASSERT_EQ(s->Peek(2, buf), 2);
  ASSERT_EQ(buf[0], 'f');
  ASSERT_EQ(buf[1], 'g');
  ASSERT_EQ(s->Peek(3, buf), 3);
  ASSERT_EQ(buf[0], 'f');
  ASSERT_EQ(buf[1], 'g');
  ASSERT_EQ(buf[2], 'h');

  for (int c = s->Get(); c != 'h'; c = s->Get()) {
  }
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'i');
  ASSERT_EQ(s->Peek(5, buf), 1);
  ASSERT_EQ(buf[0], 'i');
}

TEST_F(TestWrapperWindowsTest, TestIFStreamLessDataThanTriplePageSize) {
  // Data is 15 bytes, page size is 6 bytes, we'll cross 2 page boundaries.
  bazel::windows::AutoHandle h(FopenContents(WLINE, "abcdefghijklmno"));
  std::unique_ptr<IFStream> s(TestOnly_CreateIFStream(h, 6));
  uint8_t buf[12];

  // Move near the first page boundary.
  for (int c = s->Get(); c != 'e'; c = s->Get()) {
  }
  ASSERT_EQ(s->Peek(100, buf), 7);
  ASSERT_EQ(buf[0], 'f');
  ASSERT_EQ(buf[1], 'g');
  ASSERT_EQ(buf[2], 'h');
  ASSERT_EQ(buf[3], 'i');
  ASSERT_EQ(buf[4], 'j');
  ASSERT_EQ(buf[5], 'k');
  ASSERT_EQ(buf[6], 'l');

  // Last read character is "k".
  // Peek(2) and Peek(3) will need to read from last page.
  for (int c = s->Get(); c != 'k'; c = s->Get()) {
  }
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'l');
  ASSERT_EQ(s->Peek(100, buf), 4);
  ASSERT_EQ(buf[0], 'l');
  ASSERT_EQ(buf[1], 'm');
  ASSERT_EQ(buf[2], 'n');
  ASSERT_EQ(buf[3], 'o');

  // Move near the end of the last page.
  for (int c = s->Get(); c != 'm'; c = s->Get()) {
  }
  ASSERT_EQ(s->Peek(1, buf), 1);
  ASSERT_EQ(buf[0], 'n');
  ASSERT_EQ(s->Peek(100, buf), 2);
  ASSERT_EQ(buf[0], 'n');
  ASSERT_EQ(buf[1], 'o');
}

}  // namespace
