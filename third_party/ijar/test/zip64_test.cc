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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/zip.h"

namespace devtools_ijar {
namespace {

std::string GetTempPath(const std::string& name) {
  const char* tmpdir = getenv("TEST_TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }
  return std::string(tmpdir) + "/" + name;
}

void CreateSparseFileWithMarkers(const std::string& path, uint64_t size,
                                 uint8_t start_marker, uint8_t end_marker) {
#if defined(_WIN32)
  HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(h, INVALID_HANDLE_VALUE);
  DWORD bytes_returned = 0;
  DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &bytes_returned,
                  nullptr);
  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(size);
  ASSERT_NE(0, SetFilePointerEx(h, li, nullptr, FILE_BEGIN));
  ASSERT_NE(0, SetEndOfFile(h));
  if (size > 0) {
    li.QuadPart = 0;
    ASSERT_NE(0, SetFilePointerEx(h, li, nullptr, FILE_BEGIN));
    DWORD written = 0;
    ASSERT_NE(0, WriteFile(h, &start_marker, 1, &written, nullptr));
    ASSERT_EQ(1u, written);
    li.QuadPart = static_cast<LONGLONG>(size - 1);
    ASSERT_NE(0, SetFilePointerEx(h, li, nullptr, FILE_BEGIN));
    ASSERT_NE(0, WriteFile(h, &end_marker, 1, &written, nullptr));
    ASSERT_EQ(1u, written);
  }
  ASSERT_NE(0, CloseHandle(h));
#else
  int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  ASSERT_GE(fd, 0) << strerror(errno);
  ASSERT_EQ(0, ftruncate(fd, static_cast<off_t>(size))) << strerror(errno);
  if (size > 0) {
    ASSERT_EQ(1, pwrite(fd, &start_marker, 1, 0)) << strerror(errno);
    ASSERT_EQ(1, pwrite(fd, &end_marker, 1, static_cast<off_t>(size - 1)))
        << strerror(errno);
  }
  ASSERT_EQ(0, close(fd));
#endif
}

struct ExtractedEntry {
  std::string filename;
  uint32_t attr;
  size_t size;
  uint8_t first_byte;
  uint8_t last_byte;
};

class RecordingExtractorProcessor : public ZipExtractorProcessor {
 public:
  bool Accept(const char* filename, const u4 attr) override { return true; }

  void Process(const char* filename, const u4 attr, const u1* data,
               const size_t size) override {
    std::string_view content(reinterpret_cast<const char*>(data), size);
    ExtractedEntry entry;
    entry.filename = filename;
    entry.attr = attr;
    entry.size = size;
    entry.first_byte =
        content.empty() ? 0 : static_cast<uint8_t>(content.front());
    entry.last_byte =
        content.empty() ? 0 : static_cast<uint8_t>(content.back());
    entries.push_back(entry);
  }

  std::vector<ExtractedEntry> entries;
};

TEST(Zip64Test, ArchiveLargerThan4GiBWithMultipleFiles) {
  std::string file1 = GetTempPath("multi_file1.bin");
  std::string file2 = GetTempPath("multi_file2.bin");
  std::string file3 = GetTempPath("multi_file3.txt");
  std::string zip_path = GetTempPath("multi_4g.zip");

  constexpr uint64_t kSize1 = 2500ULL * 1024 * 1024;  // 2.5 GiB
  constexpr uint64_t kSize2 = 2000ULL * 1024 * 1024;  // 2.0 GiB
  CreateSparseFileWithMarkers(file1, kSize1, 0x11, 0x22);
  CreateSparseFileWithMarkers(file2, kSize2, 0x33, 0x44);
  CreateSparseFileWithMarkers(file3, 64, 'A', 'Z');

  const char* files[] = {file1.c_str(), file2.c_str(), file3.c_str(), nullptr};
  const char* zip_paths[] = {"file1.bin", "file2.bin", "file3.txt", nullptr};

  u8 estimated_size = ZipBuilder::EstimateSize(files, zip_paths, 3);
  ASSERT_GT(estimated_size, 0xFFFFFFFFULL);

  std::unique_ptr<ZipBuilder> builder(
      ZipBuilder::Create(zip_path.c_str(), estimated_size));
  ASSERT_NE(builder, nullptr);
  EXPECT_EQ(0, builder->AddFile("file1.bin", file1.c_str(), 0100644 << 16,
                                /*compress=*/false, /*compute_crc=*/true));
  EXPECT_EQ(0, builder->AddFile("file2.bin", file2.c_str(), 0100644 << 16,
                                /*compress=*/false, /*compute_crc=*/true));
  EXPECT_EQ(0, builder->AddFile("file3.txt", file3.c_str(), 0100644 << 16,
                                /*compress=*/false, /*compute_crc=*/true));
  EXPECT_EQ(0, builder->Finish());

  RecordingExtractorProcessor processor;
  std::unique_ptr<ZipExtractor> extractor(
      ZipExtractor::Create(zip_path.c_str(), &processor));
  ASSERT_NE(extractor, nullptr);
  EXPECT_EQ(0, extractor->ProcessAll());

  ASSERT_EQ(3u, processor.entries.size());
  EXPECT_EQ("file1.bin", processor.entries[0].filename);
  EXPECT_EQ(kSize1, processor.entries[0].size);
  EXPECT_EQ(0x11, processor.entries[0].first_byte);
  EXPECT_EQ(0x22, processor.entries[0].last_byte);

  EXPECT_EQ("file2.bin", processor.entries[1].filename);
  EXPECT_EQ(kSize2, processor.entries[1].size);
  EXPECT_EQ(0x33, processor.entries[1].first_byte);
  EXPECT_EQ(0x44, processor.entries[1].last_byte);

  EXPECT_EQ("file3.txt", processor.entries[2].filename);
  EXPECT_EQ(64u, processor.entries[2].size);
  EXPECT_EQ('A', processor.entries[2].first_byte);
  EXPECT_EQ('Z', processor.entries[2].last_byte);

  remove(file1.c_str());
  remove(file2.c_str());
  remove(file3.c_str());
  remove(zip_path.c_str());
}

TEST(Zip64Test, SingleEntryLargerThan4GiB) {
  std::string huge_file = GetTempPath("huge_entry.bin");
  std::string trailing_file = GetTempPath("trailing.txt");
  std::string zip_path = GetTempPath("single_4g.zip");

  constexpr uint64_t kHugeSize = 0x100000100ULL;  // 4 GiB + 256 bytes
  CreateSparseFileWithMarkers(huge_file, kHugeSize, 0x55, 0xAA);
  CreateSparseFileWithMarkers(trailing_file, 32, 'B', 'E');

  const char* files[] = {huge_file.c_str(), trailing_file.c_str(), nullptr};
  const char* zip_paths[] = {"huge_entry.bin", "trailing.txt", nullptr};

  u8 estimated_size = ZipBuilder::EstimateSize(files, zip_paths, 2);
  ASSERT_GT(estimated_size, kHugeSize);

  std::unique_ptr<ZipBuilder> builder(
      ZipBuilder::Create(zip_path.c_str(), estimated_size));
  ASSERT_NE(builder, nullptr);
  EXPECT_EQ(0, builder->AddFile("huge_entry.bin", huge_file.c_str(),
                                0100644 << 16, /*compress=*/false,
                                /*compute_crc=*/true));
  EXPECT_EQ(0, builder->AddFile("trailing.txt", trailing_file.c_str(),
                                0100644 << 16, /*compress=*/false,
                                /*compute_crc=*/true));
  EXPECT_EQ(0, builder->Finish());

  RecordingExtractorProcessor processor;
  std::unique_ptr<ZipExtractor> extractor(
      ZipExtractor::Create(zip_path.c_str(), &processor));
  ASSERT_NE(extractor, nullptr);
  EXPECT_EQ(0, extractor->ProcessAll());

  ASSERT_EQ(2u, processor.entries.size());
  EXPECT_EQ("huge_entry.bin", processor.entries[0].filename);
  EXPECT_EQ(kHugeSize, processor.entries[0].size);
  EXPECT_EQ(0x55, processor.entries[0].first_byte);
  EXPECT_EQ(0xAA, processor.entries[0].last_byte);

  EXPECT_EQ("trailing.txt", processor.entries[1].filename);
  EXPECT_EQ(32u, processor.entries[1].size);
  EXPECT_EQ('B', processor.entries[1].first_byte);
  EXPECT_EQ('E', processor.entries[1].last_byte);

  remove(huge_file.c_str());
  remove(trailing_file.c_str());
  remove(zip_path.c_str());
}

}  // namespace
}  // namespace devtools_ijar
