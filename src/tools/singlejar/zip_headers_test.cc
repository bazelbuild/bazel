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

#include "googletest/include/gtest/gtest.h"

#include "src/tools/singlejar/zip_headers.h"

namespace {

const uint8_t kPoison = 0xFB;

TEST(ZipHeadersTest, LocalHeader) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));
  LH *lh = reinterpret_cast<LH *>(bytes);

  // Simple fields.
  lh->signature();
  EXPECT_TRUE(lh->is());
  lh->version(123);
  EXPECT_EQ(123, lh->version());
  lh->bit_flag(0xCAFE);
  EXPECT_EQ(0xCAFE, lh->bit_flag());
  lh->compression_method(8);
  EXPECT_EQ(8, lh->compression_method());
  lh->last_mod_file_time(0xBACD);
  EXPECT_EQ(0xBACD, lh->last_mod_file_time());
  lh->last_mod_file_date(0xCDEF);
  EXPECT_EQ(0xCDEF, lh->last_mod_file_date());
  lh->crc32(0xEF015423);
  EXPECT_EQ(0xEF015423, lh->crc32());
  lh->compressed_file_size32(1234);
  EXPECT_EQ(1234UL, lh->compressed_file_size32());
  EXPECT_EQ(1234UL, lh->compressed_file_size());
  lh->uncompressed_file_size32(3421);
  EXPECT_EQ(3421UL, lh->uncompressed_file_size32());
  EXPECT_EQ(3421UL, lh->uncompressed_file_size());
  lh->file_name("foobar", 6);
  EXPECT_EQ(6UL, lh->file_name_length());
  EXPECT_EQ(0, strncmp("foobar", lh->file_name(), 6));
  EXPECT_EQ("foobar", lh->file_name_string());
  EXPECT_TRUE(lh->file_name_is("foobar"));
  lh->extra_fields(nullptr, 0);
  EXPECT_EQ(0UL, lh->extra_fields_length());
  EXPECT_EQ(30UL + 6UL, lh->size());

  // Extra fields should not be written yet
  EXPECT_EQ(kPoison, *lh->extra_fields());

  // Now copy extra fields and check we can locate them.
  // The array extra_data contains two extra fields: a 'UT' field with Unix time
  // attributes, and a Zip64 extension with two uint64 values: uncompressed file
  // size 5000000000 (0x12A05F200) and compressed file size 3000000(0x2DC6C0).
  uint8_t extra_data[] = {
    // 'UT' field: 9 bytes of payload.
    'U', 'T', 9, 0,    // tag 0x5455, length 0x0009
        0x03, 0x85, 0x0a, 0x91, 0x57, 0x7d, 0x0a, 0x91, 0x57,
    // Zip64 extension: 16 bytes of payload.
    1, 0, 16, 0,       // tag 0x0001, length 0x0010
    0, 0xf2, 0x5, 0x2a, 1, 0, 0, 0,
    0xc0, 0xc6, 0x2d, 0, 0, 0, 0, 0,
  };
  lh->extra_fields(extra_data, sizeof(extra_data));
  EXPECT_EQ(sizeof(extra_data), lh->extra_fields_length());
  EXPECT_EQ(36 + sizeof(extra_data), lh->size());
  const Zip64ExtraField *zip64_field = lh->zip64_extra_field();
  ASSERT_NE(nullptr, zip64_field);
  EXPECT_EQ(16, zip64_field->payload_size());
  EXPECT_EQ(20, zip64_field->size());
  EXPECT_EQ(5000000000UL, zip64_field->attr64(0));
  EXPECT_EQ(3000000UL, zip64_field->attr64(1));

  const UnixTimeExtraField *ut_extra_field = lh->unix_time_extra_field();
  ASSERT_NE(nullptr, ut_extra_field);
  EXPECT_EQ(9, ut_extra_field->payload_size());
  EXPECT_EQ(13, ut_extra_field->size());
  EXPECT_EQ(2, ut_extra_field->timestamp_count());
  EXPECT_TRUE(ut_extra_field->has_modification_time());
  EXPECT_TRUE(ut_extra_field->has_access_time());
  EXPECT_FALSE(ut_extra_field->has_creation_time());

  // Check that 64-bit sizes are returned correctly.
  lh->compressed_file_size32(0xFFFFFFFF);
  lh->uncompressed_file_size32(0xFFFFFFFF);
  EXPECT_EQ(3000000UL, lh->compressed_file_size());
  EXPECT_EQ(5000000000UL, lh->uncompressed_file_size());
  EXPECT_EQ(3000000UL, lh->in_zip_size());

  // Data hasn't been written:
  EXPECT_EQ(kPoison, *lh->data());
}

TEST(ZipHeadersTest, CentralDirectoryHeader) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));
  CDH *cdh = reinterpret_cast<CDH *>(bytes);

  // Simple fields.
  cdh->signature();
  EXPECT_TRUE(cdh->is());
  cdh->version(123);
  EXPECT_EQ(123, cdh->version());
  cdh->version_to_extract(321);
  EXPECT_EQ(321, cdh->version_to_extract());
  cdh->bit_flag(0xCAFE);
  EXPECT_EQ(0xCAFE, cdh->bit_flag());
  cdh->compression_method(8);
  EXPECT_EQ(8, cdh->compression_method());
  cdh->last_mod_file_time(0xBACD);
  EXPECT_EQ(0xBACD, cdh->last_mod_file_time());
  cdh->last_mod_file_date(0xCDEF);
  EXPECT_EQ(0xCDEF, cdh->last_mod_file_date());
  cdh->crc32(0xEF015423);
  EXPECT_EQ(0xEF015423, cdh->crc32());
  cdh->compressed_file_size32(1234);
  EXPECT_EQ(1234U, cdh->compressed_file_size32());
  EXPECT_EQ(1234UL, cdh->compressed_file_size());
  cdh->uncompressed_file_size32(3421);
  EXPECT_EQ(3421U, cdh->uncompressed_file_size32());
  EXPECT_EQ(3421UL, cdh->uncompressed_file_size());
  cdh->file_name("foobar", 6);
  EXPECT_EQ(6, cdh->file_name_length());
  EXPECT_EQ(0, strncmp("foobar", cdh->file_name(), 6));
  EXPECT_EQ("foobar", cdh->file_name_string());
  EXPECT_TRUE(cdh->file_name_is("foobar"));
  cdh->extra_fields(nullptr, 0);
  EXPECT_EQ(0, cdh->extra_fields_length());
  cdh->comment_length(0);
  EXPECT_EQ(0, cdh->comment_length());
  cdh->start_disk_nr(42);
  EXPECT_EQ(42, cdh->start_disk_nr());
  cdh->internal_attributes(1932);
  EXPECT_EQ(1932, cdh->internal_attributes());
  cdh->external_attributes(1234567);
  EXPECT_EQ(1234567UL, cdh->external_attributes());
  cdh->local_header_offset32(76234);
  EXPECT_EQ(76234U, cdh->local_header_offset32());
  EXPECT_EQ(76234UL, cdh->local_header_offset());

  // Only one variable field (filename) is present:
  EXPECT_EQ(46UL + 6UL, cdh->size());

  // Extra fields should not be written yet
  EXPECT_EQ(kPoison, *cdh->extra_fields());

  // Now copy extra fields and check we can locate them.
  // The array extra_data contains two extra fields: a 'UT' field with Unix time
  // attributes, and a Zip64 extension with two uint64 values: original file
  // size 5000000000 (0x12A05F200) and compressed file size 3000000(0x2DC6C0).
  uint8_t extra_data[] = {
    // 'UT' field: 5 bytes of payload (only mod time, reagrdless of flag bits)
    'U', 'T', 5, 0, 0x03, 0x85, 0x0a, 0x91, 0x57,
    // Zip64 extension: 16 bytes of payload.
    1, 0, 16, 0,    // tag 0x0001, length 0x0010
    0, 0xf2, 0x5, 0x2a, 1, 0, 0, 0,   // 0x12a05f200 = 5000000000
    0xc0, 0xc6, 0x2d, 0, 0, 0, 0, 0,  // 0x2dc6c0 = 3000000
  };
  cdh->extra_fields(extra_data, sizeof(extra_data));
  EXPECT_EQ(sizeof(extra_data), cdh->extra_fields_length());
  EXPECT_EQ(52 + sizeof(extra_data), cdh->size());
  const Zip64ExtraField *zip64_field = cdh->zip64_extra_field();
  ASSERT_NE(nullptr, zip64_field);
  EXPECT_EQ(16, zip64_field->payload_size());
  EXPECT_EQ(20, zip64_field->size());
  EXPECT_EQ(5000000000UL, zip64_field->attr64(0));
  EXPECT_EQ(3000000UL, zip64_field->attr64(1));
  const UnixTimeExtraField *ut_extra_field = cdh->unix_time_extra_field();
  ASSERT_NE(nullptr, ut_extra_field);
  EXPECT_EQ(5, ut_extra_field->payload_size());
  EXPECT_EQ(9, ut_extra_field->size());
  EXPECT_EQ(1, ut_extra_field->timestamp_count());
  EXPECT_TRUE(ut_extra_field->has_modification_time());
  EXPECT_EQ(0x57910A85UL, ut_extra_field->timestamp(0));

  // Check that 64-bit sizes are returned correctly.
  cdh->compressed_file_size32(0xFFFFFFFF);
  cdh->uncompressed_file_size32(0xFFFFFFFF);
  EXPECT_EQ(3000000UL, cdh->compressed_file_size());
  EXPECT_EQ(5000000000UL, cdh->uncompressed_file_size());

  // Check that a file with 32-bit compressed size, 64-bit original size
  // and 64-bit local header offset is handled correctly. Zip64 extension
  // field is this case contains two 64-bit quantities, original file size
  // and offset.
  uint8_t extra_data2[] = {
    1, 0, 16, 0,    // tag 0x0001, length 0x0010
    0, 0xf2, 0x5, 0x2a, 1, 0, 0, 0,   // 0x12a05f200 = 5000000000
    0, 0xbc, 0xa0, 0x65, 1, 0, 0, 0,  // 0x165A0BC00 = 6000000000
  };
  cdh->extra_fields(extra_data2, sizeof(extra_data2));
  EXPECT_EQ(sizeof(extra_data2), cdh->extra_fields_length());
  cdh->compressed_file_size32(1234);
  cdh->uncompressed_file_size32(0xFFFFFFFF);
  cdh->local_header_offset32(0xFFFFFFFF);
  EXPECT_EQ(1234UL, cdh->compressed_file_size());
  EXPECT_EQ(5000000000UL, cdh->uncompressed_file_size());
  EXPECT_EQ(6000000000UL, cdh->local_header_offset());

  // Only uncompressed file size is 64-bit quantity.
  uint8_t extra_data3[] = {
    1, 0, 8, 0,    // tag 0x0001, length 0x0008
    0, 0xbc, 0xa0, 0x65, 1, 0, 0, 0,  // 0x165A0BC00 = 6000000000
  };
  cdh->extra_fields(extra_data3, sizeof(extra_data3));
  EXPECT_EQ(sizeof(extra_data3), cdh->extra_fields_length());
  cdh->compressed_file_size32(123);
  cdh->uncompressed_file_size32(0xFFFFFFFF);
  cdh->local_header_offset32(42);
  EXPECT_EQ(123UL, cdh->compressed_file_size());
  EXPECT_EQ(6000000000UL, cdh->uncompressed_file_size());
  EXPECT_EQ(42UL, cdh->local_header_offset());
}

TEST(ZipHeadersTest, ECD64Locator) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));
  ECD64Locator *ecd64loc = reinterpret_cast<ECD64Locator *>(bytes);

  ecd64loc->signature();
  EXPECT_TRUE(ecd64loc->is());
  ecd64loc->ecd64_disk_nr(123456);
  EXPECT_EQ(123456UL, ecd64loc->ecd64_disk_nr());
  ecd64loc->ecd64_offset(6000000000);
  EXPECT_EQ(6000000000UL, ecd64loc->ecd64_offset());
  ecd64loc->total_disks(213456);
  EXPECT_EQ(213456UL, ecd64loc->total_disks());
}

TEST(ZipHeadersTest, Zip64EndOfCentralDirectory) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));

  ECD64 *ecd64 = reinterpret_cast<ECD64 *>(bytes);
  ecd64->signature();
  EXPECT_TRUE(ecd64->is());
  ecd64->remaining_size(44);
  EXPECT_EQ(44UL, ecd64->remaining_size());
  ecd64->version(56);
  EXPECT_EQ(56UL, ecd64->version());
  ecd64->version_to_extract(754);
  EXPECT_EQ(754UL, ecd64->version_to_extract());
  ecd64->this_disk_nr(75123);
  EXPECT_EQ(75123UL, ecd64->this_disk_nr());
  ecd64->cen_disk_nr(87654);
  EXPECT_EQ(87654UL, ecd64->cen_disk_nr());
  ecd64->this_disk_entries(9000000000);
  EXPECT_EQ(9000000000UL, ecd64->this_disk_entries());
  ecd64->total_entries(8000000000);
  EXPECT_EQ(8000000000UL, ecd64->total_entries());
  ecd64->cen_size(19000000000);
  EXPECT_EQ(19000000000UL, ecd64->cen_size());
  ecd64->cen_offset(11000000000);
  EXPECT_EQ(11000000000UL, ecd64->cen_offset());
}

TEST(ZipHeadersTest, EndOfCentralDirectory) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));
  ECD64Locator *ecd64loc = reinterpret_cast<ECD64Locator *>(bytes);
  ECD *ecd = reinterpret_cast<ECD *>(bytes + sizeof(ECD64Locator));

  ecd->signature();
  EXPECT_TRUE(ecd->is());
  ecd->this_disk_nr(123);
  EXPECT_EQ(123, ecd->this_disk_nr());
  ecd->cen_disk_nr(4123);
  EXPECT_EQ(4123, ecd->cen_disk_nr());
  ecd->this_disk_entries16(23);
  EXPECT_EQ(23, ecd->this_disk_entries16());
  ecd->total_entries16(23);
  EXPECT_EQ(23, ecd->total_entries16());
  ecd->cen_size32(123400);
  EXPECT_EQ(123400U, ecd->cen_size32());
  ecd->cen_offset32(123000);
  EXPECT_EQ(123000U, ecd->cen_offset32());
  uint8_t comment_bytes[] = {0xCA, 0xFE};
  ecd->comment(comment_bytes, sizeof(comment_bytes));
  EXPECT_EQ(sizeof(comment_bytes), ecd->comment_length());
  EXPECT_EQ(comment_bytes[0], ecd->comment()[0]);
  EXPECT_EQ(comment_bytes[1], ecd->comment()[1]);

  // ECD64 locator has not been constructed yet, so:
  EXPECT_EQ(0xFFFFFFFFFFFFFFFFLL, ecd->ecd64_offset());

  // Now construct it and see it used:
  ecd64loc->signature();
  ecd64loc->ecd64_offset(9876543210);
  EXPECT_EQ(9876543210UL, ecd->ecd64_offset());
}

TEST(ZipHeadersTest, Zip64ExtraFieldTest) {
  uint8_t bytes[256];
  memset(bytes, kPoison, sizeof(bytes));
  Zip64ExtraField *z64 = reinterpret_cast<Zip64ExtraField *>(bytes);

  z64->signature();
  EXPECT_TRUE(z64->is());
  z64->payload_size(16);
  z64->attr64(0, 9876543210);
  EXPECT_EQ(9876543210UL, z64->attr64(0));
  z64->attr64(1, 8976543210);
  EXPECT_EQ(8976543210UL, z64->attr64(1));
  EXPECT_EQ(kPoison, bytes[z64->size()]);
}

}  // namespace

