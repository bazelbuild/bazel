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
#include <string>
#include <vector>

#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "src/tools/singlejar/zip_headers.h"
#include "googletest/include/gtest/gtest.h"

static const char kJar[] = "jar.jar";

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

TEST(InputJarBadJarTest, OpenInMemoryTooSmall) {
  std::vector<uint8_t> buf(10, 0);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, ECDCommentTooLong) {
  std::vector<uint8_t> buf(sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data());
  ecd->signature();
  uint8_t comment[] = {'a', 'b'};
  // Set comment_length to 10 without space in buffer.
  ecd->comment(comment, 0);
  buf[sizeof(ECD) - 2] = 10;
  buf[sizeof(ECD) - 1] = 0;
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, TruncatedBeforeECD64Locator) {
  // File has 2 bytes + ECD (22 bytes) = 24 bytes, so offset(ecd) < 20.
  std::vector<uint8_t> buf(2 + sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + 2);
  ecd->signature();
  ecd->cen_size32(1);
  ecd->cen_offset32(0);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, TruncatedBeforeECD64) {
  // File has 2 bytes + ECD64Locator (20 bytes) + ECD (22 bytes) = 44 bytes,
  // so offset(ecd64loc) == 2 < sizeof(ECD64) (56 bytes).
  std::vector<uint8_t> buf(2 + sizeof(ECD64Locator) + sizeof(ECD), 0);
  auto* loc = reinterpret_cast<ECD64Locator*>(buf.data() + 2);
  loc->signature();
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + 2 + sizeof(ECD64Locator));
  ecd->signature();
  ecd->cen_size32(0xFFFFFFFF);
  ecd->cen_offset32(0xFFFFFFFF);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, CentralDirectorySizeTooLarge64) {
  std::vector<uint8_t> buf(sizeof(ECD64) + sizeof(ECD64Locator) + sizeof(ECD),
                           0);
  auto* ecd64 = reinterpret_cast<ECD64*>(buf.data());
  ecd64->signature();
  ecd64->cen_size(1000);
  ecd64->cen_offset(0);
  auto* loc = reinterpret_cast<ECD64Locator*>(buf.data() + sizeof(ECD64));
  loc->signature();
  auto* ecd =
      reinterpret_cast<ECD*>(buf.data() + sizeof(ECD64) + sizeof(ECD64Locator));
  ecd->signature();
  ecd->cen_size32(0xFFFFFFFF);
  ecd->cen_offset32(0xFFFFFFFF);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, CentralDirectoryOffsetExceedsPosition32) {
  // 100 bytes before ECD, cen_size = 50 (so cdh_ is at offset 50),
  // but cen_offset32 = 80 (> 50), which would underflow preamble_size_.
  std::vector<uint8_t> buf(100 + sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + 100);
  ecd->signature();
  ecd->cen_size32(50);
  ecd->cen_offset32(80);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, CentralDirectoryOffsetExceedsPosition64) {
  std::vector<uint8_t> buf(
      100 + sizeof(ECD64) + sizeof(ECD64Locator) + sizeof(ECD), 0);
  auto* ecd64 = reinterpret_cast<ECD64*>(buf.data() + 100);
  ecd64->signature();
  ecd64->cen_size(50);
  ecd64->cen_offset(80);
  auto* loc = reinterpret_cast<ECD64Locator*>(buf.data() + 100 + sizeof(ECD64));
  loc->signature();
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + 100 + sizeof(ECD64) +
                                     sizeof(ECD64Locator));
  ecd->signature();
  ecd->cen_size32(0xFFFFFFFF);
  ecd->cen_offset32(0xFFFFFFFF);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, TruncatedFirstCDH) {
  // Central directory has size 10 (< sizeof(CDH) == 46), starting with CDH sig.
  std::vector<uint8_t> buf(10 + sizeof(ECD), 0);
  buf[0] = 0x50;
  buf[1] = 0x4b;
  buf[2] = 0x01;
  buf[3] = 0x02;
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + 10);
  ecd->signature();
  ecd->cen_size32(10);
  ecd->cen_offset32(0);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, FirstCDHBadSignature) {
  // Central directory has sizeof(CDH) bytes, but does not start with CDH sig.
  std::vector<uint8_t> buf(sizeof(CDH) + sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(0);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, EmptyCentralDirectoryWithExt64Offset) {
  std::vector<uint8_t> buf(sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data());
  ecd->signature();
  ecd->cen_size32(0);
  ecd->cen_offset32(0xFFFFFFFF);
  InputJar input_jar;
  EXPECT_FALSE(input_jar.Open(kJar, buf.data(), buf.size()));
}

TEST(InputJarBadJarTest, CDHSizeExceedsCentralDirectory) {
  // Central directory has 46 bytes (sizeof(CDH)), but CDH claims
  // file_name_length = 10, so cdh->size() = 56 > cen_size (46).
  std::vector<uint8_t> buf(sizeof(CDH) + sizeof(ECD), 0);
  auto* cdh = reinterpret_cast<CDH*>(buf.data());
  cdh->signature();
  cdh->file_name("a", 0);
  // Overwrite file_name_length_ (at offset 28 of CDH) to 10 without writing
  // past CDH.
  buf[28] = 10;
  buf[29] = 0;
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(0);
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad directory record");
}

TEST(InputJarBadJarTest, TruncatedSecondCDH) {
  // Valid first LH + CDH, followed by 2 trailing bytes in the central
  // directory (< sizeof(CDH)) before ECD.
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + 2 + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->local_header_offset32(0);
  auto* ecd =
      reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH) + 2);
  ecd->signature();
  ecd->cen_size32(sizeof(CDH) + 2);
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_NE(nullptr, input_jar.NextEntry(&lh));
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad directory record");
}

TEST(InputJarBadJarTest, SecondCDHBadSignature) {
  // Valid first LH + CDH, followed by a second CDH-sized region with an
  // invalid signature before ECD.
  std::vector<uint8_t> buf(sizeof(LH) + 2 * sizeof(CDH) + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->local_header_offset32(0);
  auto* ecd =
      reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + 2 * sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(2 * sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_NE(nullptr, input_jar.NextEntry(&lh));
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad directory record");
}

TEST(InputJarBadJarTest, LocalHeaderOffsetOutOfBounds) {
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->local_header_offset32(0x10000);  // Out of bounds
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad local header offset");
}

TEST(InputJarBadJarTest, LocalHeaderBadSignature) {
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + sizeof(ECD), 0);
  // LH at offset 0 does NOT have valid signature.
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->local_header_offset32(0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad local header");
}

TEST(InputJarBadJarTest, LocalHeaderSizeOutOfBounds) {
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  // Set LH file_name_length_ (at offset 26 of LH) to 1000 (past EOF).
  buf[26] = 0xe8;
  buf[27] = 0x03;
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->local_header_offset32(0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad local header");
}

TEST(InputJarBadJarTest, EntrySizeOutOfBounds) {
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  lh_ptr->compression_method(0);
  lh_ptr->compressed_file_size32(0x10000);
  lh_ptr->uncompressed_file_size32(0x10000);
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->compression_method(0);
  cdh->compressed_file_size32(0x10000);
  cdh->uncompressed_file_size32(0x10000);
  cdh->local_header_offset32(0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
  const LH* lh = nullptr;
  EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad entry size");
}

TEST(InputJarBadJarTest, DataDescriptorOutOfBounds) {
  std::vector<uint8_t> buf(sizeof(LH) + sizeof(CDH) + sizeof(ECD), 0);
  auto* lh_ptr = reinterpret_cast<LH*>(buf.data());
  lh_ptr->signature();
  lh_ptr->bit_flag(0x08);
  lh_ptr->compression_method(8);
  auto* cdh = reinterpret_cast<CDH*>(buf.data() + sizeof(LH));
  cdh->signature();
  cdh->bit_flag(0x08);  // no_size_in_local_header() == true
  cdh->compression_method(8);
  // Set compressed_file_size32 so less than sizeof(DDR) remains in
  // mapped_file_.
  cdh->compressed_file_size32(sizeof(CDH) + sizeof(ECD) - 2);
  cdh->uncompressed_file_size32(10);
  cdh->local_header_offset32(0);
  auto* ecd = reinterpret_cast<ECD*>(buf.data() + sizeof(LH) + sizeof(CDH));
  ecd->signature();
  ecd->cen_size32(sizeof(CDH));
  ecd->cen_offset32(sizeof(LH));
  {
    InputJar input_jar;
    ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
    const LH* lh = nullptr;
    EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad entry size");
  }

  // Set compressed_file_size32 so sizeof(DDR) (4 bytes) remains in
  // mapped_file_, which is less than the full data descriptor size (12 bytes).
  cdh->compressed_file_size32(sizeof(CDH) + sizeof(ECD) - sizeof(DDR));
  {
    InputJar input_jar;
    ASSERT_TRUE(input_jar.Open(kJar, buf.data(), buf.size()));
    const LH* lh = nullptr;
    EXPECT_DEATH(input_jar.NextEntry(&lh), "Bad data descriptor");
  }
}
