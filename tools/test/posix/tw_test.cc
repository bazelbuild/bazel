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

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/zip.h"
#include "tools/test/posix/tw_outputs.h"
#include "tools/test/test_wrapper_common.h"

namespace bazel {
namespace tools {
namespace test_wrapper {
namespace {

struct ExtractedEntry {
  std::string content;
  bool is_directory;
};

class InMemoryZipExtractor : public devtools_ijar::ZipExtractorProcessor {
 public:
  bool Accept(const char* filename, devtools_ijar::u4 attr) override {
    return true;
  }

  void Process(const char* filename, devtools_ijar::u4 attr,
               const devtools_ijar::u1* data, size_t size) override {
    ExtractedEntry entry;
    if (size != 0) {
      entry.content.assign(reinterpret_cast<const char*>(data), size);
    }
    entry.is_directory = devtools_ijar::zipattr_is_dir(attr);
    entries_[filename] = entry;
  }

  const std::map<std::string, ExtractedEntry>& entries() const {
    return entries_;
  }

 private:
  std::map<std::string, ExtractedEntry> entries_;
};

class TestWrapperPosixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* test_tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(test_tmpdir, nullptr);

    std::string pattern =
        std::string(test_tmpdir) + "/posix-test-wrapper.XXXXXX";
    std::vector<char> writable_pattern(pattern.begin(), pattern.end());
    writable_pattern.push_back('\0');
    char* directory = mkdtemp(writable_pattern.data());
    ASSERT_NE(directory, nullptr) << std::strerror(errno);
    directory_ = directory;
    root_ = directory_ + "/outputs";
    MakeDirectory(root_);
  }

  void TearDown() override {
    if (!directory_.empty()) {
      EXPECT_TRUE(blaze_util::RemoveRecursively(blaze_util::Path(directory_)))
          << directory_ << ": " << std::strerror(errno);
    }
  }

  void MakeDirectory(const std::string& path) {
    ASSERT_EQ(mkdir(path.c_str(), 0755), 0)
        << path << ": " << std::strerror(errno);
  }

  void WriteFile(const std::string& path, const std::string& content) {
    std::ofstream stream(path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(stream.is_open()) << path << ": " << std::strerror(errno);
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.close();
    ASSERT_FALSE(stream.fail()) << path;
  }

  std::string ReadFile(const std::string& path) {
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream.is_open()) {
      ADD_FAILURE() << path << ": " << std::strerror(errno);
      return "";
    }
    std::ostringstream result;
    result << stream.rdbuf();
    if (stream.bad()) {
      ADD_FAILURE() << "Failed to read " << path;
      return "";
    }
    return result.str();
  }

  bool Exists(const std::string& path) const {
    struct stat file_stat;
    return lstat(path.c_str(), &file_stat) == 0;
  }

  void ExtractZip(const std::string& path,
                  InMemoryZipExtractor* extracted_entries) {
    std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
        devtools_ijar::ZipExtractor::Create(path.c_str(), extracted_entries));
    ASSERT_NE(extractor, nullptr) << path << ": " << std::strerror(errno);
    ASSERT_EQ(extractor->ProcessAll(), 0)
        << (extractor->GetError() == nullptr ? "Unknown ZIP error"
                                             : extractor->GetError());
  }

  std::string directory_;
  std::string root_;
};

TEST(TestWrapperCommonTest, EmptyZipEntryPathsAreNullTerminated) {
  ZipEntryPaths paths;
  paths.Create("/execution/root", {});

  EXPECT_EQ(paths.Size(), 0);
  ASSERT_NE(paths.AbsPathPtrs(), nullptr);
  ASSERT_NE(paths.EntryPathPtrs(), nullptr);
  EXPECT_EQ(paths.AbsPathPtrs()[0], nullptr);
  EXPECT_EQ(paths.EntryPathPtrs()[0], nullptr);
}

TEST(TestWrapperCommonTest, ZipEntryPathsPreserveDirectoriesAndSpaces) {
  ZipEntryPaths paths;
  paths.Create("/execution/root", {"directory/", "directory/a file.txt"});

  ASSERT_EQ(paths.Size(), 2);
  EXPECT_STREQ(paths.AbsPathPtrs()[0], "/execution/root/directory/");
  EXPECT_STREQ(paths.AbsPathPtrs()[1], "/execution/root/directory/a file.txt");
  EXPECT_EQ(paths.AbsPathPtrs()[2], nullptr);
  EXPECT_STREQ(paths.EntryPathPtrs()[0], "directory/");
  EXPECT_STREQ(paths.EntryPathPtrs()[1], "directory/a file.txt");
  EXPECT_EQ(paths.EntryPathPtrs()[2], nullptr);

  paths.Create("/other", {"replacement"});
  ASSERT_EQ(paths.Size(), 1);
  EXPECT_STREQ(paths.AbsPathPtrs()[0], "/other/replacement");
  EXPECT_STREQ(paths.EntryPathPtrs()[0], "replacement");
  EXPECT_EQ(paths.AbsPathPtrs()[1], nullptr);
  EXPECT_EQ(paths.EntryPathPtrs()[1], nullptr);
}

TEST(TestWrapperCommonTest, ManifestFormattingSupportsFullUint64Range) {
  EXPECT_EQ(
      FormatUndeclaredOutputManifestEntry("nested/file.txt", 10, "text/plain"),
      "nested/file.txt\t10\ttext/plain\n");
  EXPECT_EQ(FormatUndeclaredOutputManifestEntry(
                "large.bin", std::numeric_limits<uint64_t>::max(),
                "application/octet-stream"),
            "large.bin\t18446744073709551615\tapplication/octet-stream\n");
}

TEST_F(TestWrapperPosixTest, ManifestIsSortedAndPreservesUnarchivedOutputs) {
  MakeDirectory(root_ + "/nested");
  WriteFile(root_ + "/zeta", "last\n");
  WriteFile(root_ + "/nested/index.html", "<!DOCTYPE html>\n");
  WriteFile(root_ + "/alpha.txt", "some text\n");
  WriteFile(root_ + "/.hidden", std::string("\0\1\xff", 3));

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.manifest = directory_ + "/metadata/nested/MANIFEST";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.manifest),
            ".hidden\t3\tapplication/octet-stream\n"
            "alpha.txt\t10\ttext/plain\n"
            "nested/index.html\t16\ttext/html\n"
            "zeta\t5\ttext/plain\n");
  EXPECT_TRUE(Exists(root_ + "/.hidden"));
  EXPECT_TRUE(Exists(root_ + "/alpha.txt"));
  EXPECT_TRUE(Exists(root_ + "/nested/index.html"));
  EXPECT_TRUE(Exists(root_ + "/zeta"));
}

TEST_F(TestWrapperPosixTest, MimeTypeUsesContentInsteadOfFilename) {
  WriteFile(root_ + "/binary.txt", std::string("\0\1\2", 3));
  WriteFile(root_ + "/document.bin", "<HTML><body>content</body></HTML>\n");
  WriteFile(root_ + "/empty.txt", "");
  WriteFile(root_ + "/text.dat", "ordinary text\n");

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.manifest = directory_ + "/MANIFEST";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.manifest),
            "binary.txt\t3\tapplication/octet-stream\n"
            "document.bin\t34\ttext/html\n"
            "empty.txt\t0\tinode/x-empty\n"
            "text.dat\t14\ttext/plain\n");
}

TEST_F(TestWrapperPosixTest,
       ZipContainsDirectoriesHiddenFilesAndSortedManifest) {
  MakeDirectory(root_ + "/.secrets");
  MakeDirectory(root_ + "/empty");
  MakeDirectory(root_ + "/nested");
  MakeDirectory(root_ + "/nested/deeper");
  WriteFile(root_ + "/.hidden", "hidden\n");
  WriteFile(root_ + "/.secrets/inside.txt", "secret\n");
  WriteFile(root_ + "/nested/deeper/leaf.txt", "leaf\n");
  WriteFile(root_ + "/zeta.html", "<!DOCTYPE html>\n");

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.zip = root_ + "/outputs.zip";
  outputs.manifest = directory_ + "/manifest/MANIFEST";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.manifest),
            ".hidden\t7\ttext/plain\n"
            ".secrets/inside.txt\t7\ttext/plain\n"
            "nested/deeper/leaf.txt\t5\ttext/plain\n"
            "zeta.html\t16\ttext/html\n");
  EXPECT_TRUE(Exists(outputs.zip));
  EXPECT_FALSE(Exists(root_ + "/.hidden"));
  EXPECT_FALSE(Exists(root_ + "/.secrets"));
  EXPECT_FALSE(Exists(root_ + "/empty"));
  EXPECT_FALSE(Exists(root_ + "/nested"));
  EXPECT_FALSE(Exists(root_ + "/zeta.html"));

  InMemoryZipExtractor archive;
  ExtractZip(outputs.zip, &archive);
  const std::map<std::string, ExtractedEntry>& entries = archive.entries();
  ASSERT_EQ(entries.size(), 8);
  ASSERT_EQ(entries.count(".hidden"), 1);
  ASSERT_EQ(entries.count(".secrets/"), 1);
  ASSERT_EQ(entries.count(".secrets/inside.txt"), 1);
  ASSERT_EQ(entries.count("empty/"), 1);
  ASSERT_EQ(entries.count("nested/"), 1);
  ASSERT_EQ(entries.count("nested/deeper/"), 1);
  ASSERT_EQ(entries.count("nested/deeper/leaf.txt"), 1);
  ASSERT_EQ(entries.count("zeta.html"), 1);
  EXPECT_EQ(entries.at(".hidden").content, "hidden\n");
  EXPECT_EQ(entries.at(".secrets/inside.txt").content, "secret\n");
  EXPECT_EQ(entries.at("nested/deeper/leaf.txt").content, "leaf\n");
  EXPECT_EQ(entries.at("zeta.html").content, "<!DOCTYPE html>\n");
  EXPECT_TRUE(entries.at(".secrets/").is_directory);
  EXPECT_TRUE(entries.at("empty/").is_directory);
  EXPECT_TRUE(entries.at("nested/").is_directory);
  EXPECT_TRUE(entries.at("nested/deeper/").is_directory);
  EXPECT_FALSE(entries.at(".hidden").is_directory);
}

TEST_F(TestWrapperPosixTest, ArchiveCanBeOutsideUndeclaredOutputDirectory) {
  WriteFile(root_ + "/output.txt", "content\n");

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.zip = directory_ + "/archive/nested/outputs.zip";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_TRUE(Exists(outputs.zip));
  EXPECT_FALSE(Exists(root_ + "/output.txt"));
  InMemoryZipExtractor archive;
  ExtractZip(outputs.zip, &archive);
  ASSERT_EQ(archive.entries().size(), 1);
  EXPECT_EQ(archive.entries().at("output.txt").content, "content\n");
}

TEST_F(TestWrapperPosixTest,
       AnnotationsConcatenateSortedTopLevelPartAndProtoFiles) {
  const std::string annotation_directory = directory_ + "/annotation_parts";
  MakeDirectory(annotation_directory);
  MakeDirectory(annotation_directory + "/nested");
  MakeDirectory(annotation_directory + "/directory.part");
  WriteFile(annotation_directory + "/z.part", "last");
  WriteFile(annotation_directory + "/a.part", "first");
  WriteFile(annotation_directory + "/.hidden.part", "hidden");
  WriteFile(annotation_directory + "/ignored.txt", "ignored");
  WriteFile(annotation_directory + "/nested/nested.part", "nested");
  WriteFile(annotation_directory + "/z.pb", std::string("\0z", 2));
  WriteFile(annotation_directory + "/a.pb", std::string("\1a", 2));
  WriteFile(annotation_directory + "/.hidden.pb", "hidden");

  UndeclaredOutputs outputs;
  outputs.annotations_dir = annotation_directory;
  outputs.annotations = directory_ + "/metadata/nested/ANNOTATIONS";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.annotations), "firstlast");
  EXPECT_EQ(ReadFile(outputs.annotations + ".pb"), std::string("\1a\0z", 4));
  EXPECT_TRUE(Exists(annotation_directory + "/a.part"));
  EXPECT_TRUE(Exists(annotation_directory + "/z.pb"));
}

TEST_F(TestWrapperPosixTest, HiddenAnnotationFragmentsDoNotCreateOutputs) {
  const std::string annotation_directory = directory_ + "/annotation_parts";
  MakeDirectory(annotation_directory);
  WriteFile(annotation_directory + "/.hidden.part", "hidden");
  WriteFile(annotation_directory + "/.hidden.pb", "hidden");
  WriteFile(annotation_directory + "/ignored.txt", "ignored");

  UndeclaredOutputs outputs;
  outputs.annotations_dir = annotation_directory;
  outputs.annotations = directory_ + "/ANNOTATIONS";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_FALSE(Exists(outputs.annotations));
  EXPECT_FALSE(Exists(outputs.annotations + ".pb"));
}

TEST_F(TestWrapperPosixTest, EmptyDirectoriesDoNotCreateOutputArtifacts) {
  const std::string annotation_directory = directory_ + "/annotation_parts";
  MakeDirectory(annotation_directory);

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.zip = root_ + "/outputs.zip";
  outputs.manifest = directory_ + "/MANIFEST";
  outputs.annotations_dir = annotation_directory;
  outputs.annotations = directory_ + "/ANNOTATIONS";
  std::string error = "previous error";
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_TRUE(error.empty());
  EXPECT_FALSE(Exists(outputs.zip));
  EXPECT_FALSE(Exists(outputs.manifest));
  EXPECT_FALSE(Exists(outputs.annotations));
  EXPECT_FALSE(Exists(outputs.annotations + ".pb"));
}

TEST_F(TestWrapperPosixTest, MissingDirectoriesDoNotCreateOutputArtifacts) {
  UndeclaredOutputs outputs;
  outputs.root = directory_ + "/missing_outputs";
  outputs.zip = directory_ + "/outputs.zip";
  outputs.manifest = directory_ + "/MANIFEST";
  outputs.annotations_dir = directory_ + "/missing_annotations";
  outputs.annotations = directory_ + "/ANNOTATIONS";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_FALSE(Exists(outputs.zip));
  EXPECT_FALSE(Exists(outputs.manifest));
  EXPECT_FALSE(Exists(outputs.annotations));
  EXPECT_FALSE(Exists(outputs.annotations + ".pb"));
}

TEST_F(TestWrapperPosixTest, SymlinkTargetsAreArchivedButNotDeleted) {
  const std::string external_directory = directory_ + "/external";
  MakeDirectory(external_directory);
  WriteFile(external_directory + "/file.txt", "outside\n");
  MakeDirectory(external_directory + "/directory");
  WriteFile(external_directory + "/directory/child.txt", "nested\n");

  ASSERT_EQ(symlink((external_directory + "/file.txt").c_str(),
                    (root_ + "/file-link.txt").c_str()),
            0)
      << std::strerror(errno);
  ASSERT_EQ(symlink((external_directory + "/directory").c_str(),
                    (root_ + "/directory-link").c_str()),
            0)
      << std::strerror(errno);

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.zip = root_ + "/outputs.zip";
  outputs.manifest = directory_ + "/MANIFEST";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.manifest),
            "directory-link/child.txt\t7\ttext/plain\n"
            "file-link.txt\t8\ttext/plain\n");
  EXPECT_FALSE(Exists(root_ + "/file-link.txt"));
  EXPECT_FALSE(Exists(root_ + "/directory-link"));
  EXPECT_EQ(ReadFile(external_directory + "/file.txt"), "outside\n");
  EXPECT_EQ(ReadFile(external_directory + "/directory/child.txt"), "nested\n");

  InMemoryZipExtractor archive;
  ExtractZip(outputs.zip, &archive);
  ASSERT_EQ(archive.entries().size(), 3);
  EXPECT_TRUE(archive.entries().at("directory-link/").is_directory);
  EXPECT_EQ(archive.entries().at("directory-link/child.txt").content,
            "nested\n");
  EXPECT_EQ(archive.entries().at("file-link.txt").content, "outside\n");
}

TEST_F(TestWrapperPosixTest, SymlinkCyclesDoNotRecurseIndefinitely) {
  WriteFile(root_ + "/file.txt", "content\n");
  ASSERT_EQ(symlink(".", (root_ + "/cycle").c_str()), 0)
      << std::strerror(errno);

  UndeclaredOutputs outputs;
  outputs.root = root_;
  outputs.zip = root_ + "/outputs.zip";
  outputs.manifest = directory_ + "/MANIFEST";
  std::string error;
  ASSERT_TRUE(ProcessUndeclaredOutputs(outputs, &error)) << error;

  EXPECT_EQ(ReadFile(outputs.manifest), "file.txt\t8\ttext/plain\n");
  InMemoryZipExtractor archive;
  ExtractZip(outputs.zip, &archive);
  ASSERT_EQ(archive.entries().size(), 2);
  EXPECT_TRUE(archive.entries().at("cycle/").is_directory);
  EXPECT_EQ(archive.entries().at("file.txt").content, "content\n");
}

TEST_F(TestWrapperPosixTest, NondirectoryRootReturnsUsefulError) {
  const std::string invalid_root = directory_ + "/not_a_directory";
  WriteFile(invalid_root, "content");

  UndeclaredOutputs outputs;
  outputs.root = invalid_root;
  outputs.manifest = directory_ + "/MANIFEST";
  std::string error;
  EXPECT_FALSE(ProcessUndeclaredOutputs(outputs, &error));
  EXPECT_NE(error.find("not a directory"), std::string::npos) << error;
  EXPECT_NE(error.find(invalid_root), std::string::npos) << error;
  EXPECT_FALSE(Exists(outputs.manifest));
}

}  // namespace
}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
