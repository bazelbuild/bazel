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

#include <cstdint>
#include <cstdio>
#include <string>

#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "googletest/include/gtest/gtest.h"

static const char kJar[] = "jar.jar";

namespace {
bool WriteBytes(const std::string &path, const uint8_t *data, size_t size) {
  FILE *fp = fopen(path.c_str(), "wb");
  if (fp == nullptr) {
    return false;
  }
  bool ok = fwrite(data, 1, size, fp) == size;
  return fclose(fp) == 0 && ok;
}
}  // namespace

TEST(InputJarBadJarTest, NotAJar) {
  std::string out_path = singlejar_test_util::OutputFilePath(kJar);
  ASSERT_TRUE(singlejar_test_util::AllocateFile(out_path, 1000));
  InputJar input_jar;
  ASSERT_FALSE(input_jar.Open(out_path));
}

// Check that an empty file does not cause trouble in MappedFile.
TEST(InputJarBadJarTest, EmptyFile) {
  std::string out_path = singlejar_test_util::OutputFilePath(kJar);
  ASSERT_TRUE(singlejar_test_util::AllocateFile(out_path, 0));
  InputJar input_jar;
  ASSERT_FALSE(input_jar.Open(out_path));
}

// A crafted archive that presents a valid ECD, ECD64 Locator and ECD64 record,
// but whose ECD64 Central Directory size is far larger than the file. The
// Central Directory header pointer is computed as `ecd64 - cen_size`, so an
// unchecked size makes it point outside the mapped file. `Open()` must reject
// the archive instead of dereferencing an out-of-bounds pointer.
TEST(InputJarBadJarTest, Zip64CentralDirectorySizeOutOfBounds) {
  // Layout: [ECD64 (56 bytes)][ECD64Locator (20 bytes)][ECD (22 bytes)].
  static const uint8_t kArchive[] = {
      // ECD64: signature, remaining_size, version/version_to_extract,
      // this_disk_nr, cen_disk_nr, this_disk_entries, total_entries,
      // cen_size (0x0000000000100000), cen_offset.
      0x50, 0x4b, 0x06, 0x06, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      // ECD64Locator: signature, ecd64_disk_nr, ecd64_offset, total_disks.
      0x50, 0x4b, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      // ECD: signature, disk numbers/entry counts, cen_size32 (0xFFFFFFFF),
      // cen_offset32 (0xFFFFFFFF), comment_length. The 0xFFFFFFFF markers force
      // the ZIP64 code path.
      0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
  };
  std::string out_path = singlejar_test_util::OutputFilePath(kJar);
  ASSERT_TRUE(WriteBytes(out_path, kArchive, sizeof(kArchive)));
  InputJar input_jar;
  ASSERT_FALSE(input_jar.Open(out_path));
}
