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

#include <stdio.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>

#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "src/tools/singlejar/transient_bytes.h"
#include "googletest/include/gtest/gtest.h"

#ifdef _MSC_VER
#define SINGLEJAR_ALYWAYS_INLINE __forceinline
#else
#define SINGLEJAR_ALYWAYS_INLINE __attribute__((always_inline))
#endif

namespace {
const char kStoredJar[] = "stored.zip";
const char kCompressedJar[] = "compressed.zip";
const char kBytesSmall[] =
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789"
    "0123456789012345678901234567890123456789";

std::ostream &operator<<(std::ostream &out,
                                TransientBytes const &bytes) {
  struct Sink {
    void operator()(const void *chunk, uint64_t chunk_size) const {
      out_.write(reinterpret_cast<const char *>(chunk), chunk_size);
    }
    std::ostream &out_;
  };
  Sink sink{out};
  bytes.stream_out(sink);
  return out;
}

class TransientBytesTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    ASSERT_EQ(0, chdir(getenv("TEST_TMPDIR")));
    CreateCompressedJar();
  }

  static void TearDownTestCase() { unlink(kCompressedJar); }

  void SetUp() override { transient_bytes_.reset(new TransientBytes); }

  // The value of the byte at a given position in a file created by the
  // CreateFile method below.
  static SINGLEJAR_ALYWAYS_INLINE uint8_t file_byte_at(uint64_t offset) {
    // return offset >> (8 * (offset & 7));
    return offset & 255;
  }

  // Create file with given name and size and contents.
  static bool CreateFile(const char *filename, uint64_t size) {
    FILE *fp = fopen(filename, "wb");
    if (fp == nullptr) {
      perror(filename);
      return false;
    }
    const uint64_t buffer_size = 4096;
    uint8_t buffer[buffer_size];
    uint64_t offset = 0;
    while (offset < size) {
      uint64_t offset_end = std::min(size, offset + buffer_size);
      uint64_t to_write = 0;
      while (offset < offset_end) {
        buffer[to_write++] = file_byte_at(offset++);
      }
      if (fwrite(buffer, to_write, 1, fp) != 1) {
        perror(filename);
        fclose(fp);
        return false;
      }
    }
    if (0 == fclose(fp)) {
      return true;
    }
    perror(filename);
    return false;
  }

  static void CreateStoredJar() {
    ASSERT_TRUE(singlejar_test_util::AllocateFile("small1", 100));
    ASSERT_TRUE(singlejar_test_util::AllocateFile("huge", 0x100000001));
    ASSERT_TRUE(singlejar_test_util::AllocateFile("small2", 100));
    unlink(kStoredJar);
    ASSERT_EQ(0, system("zip -0qm stored.zip small1 huge small2"));
#if !defined(__APPLE__)
    ASSERT_EQ(0, system("unzip -v stored.zip"));
#endif
  }

  static void CreateCompressedJar() {
    unlink(kCompressedJar);
    ASSERT_TRUE(CreateFile("511", 511));
    ASSERT_TRUE(CreateFile("huge", 0x100000001));
    ASSERT_TRUE(CreateFile("1K", 1024));
    ASSERT_EQ(0, system("zip -qm compressed.zip 511 huge 1K"));
#if !defined(__APPLE__)
    ASSERT_EQ(0, system("unzip -v compressed.zip"));
#endif
  }
  std::unique_ptr<TransientBytes> transient_bytes_;
};

TEST_F(TransientBytesTest, AppendBytes) {
  int const kIter = 10000;
  transient_bytes_->Append(kBytesSmall);
  EXPECT_EQ(strlen(kBytesSmall), transient_bytes_->data_size());
  std::ostringstream out;
  out << *transient_bytes_;
  EXPECT_STREQ(kBytesSmall, out.str().c_str());
  out.flush();

  for (int i = 1; i < kIter; ++i) {
    transient_bytes_->Append(kBytesSmall);
    ASSERT_EQ((i + 1) * strlen(kBytesSmall), transient_bytes_->data_size());
  }

  out << *transient_bytes_;
  std::string out_string = out.str();
  size_t size = strlen(kBytesSmall);
  for (size_t pos = 0; pos < kIter * size; pos += size) {
    ASSERT_STREQ(kBytesSmall, out_string.substr(pos, size).c_str())
        << (pos / size) << "-th chunk does not match";
  }
}

TEST_F(TransientBytesTest, ReadEntryContents) {
  ASSERT_EQ(0, chdir(getenv("TEST_TMPDIR")));
  CreateStoredJar();
  std::unique_ptr<InputJar> input_jar(new InputJar);
  ASSERT_TRUE(input_jar->Open(kStoredJar));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar->NextEntry(&lh))) {
    transient_bytes_.reset(new TransientBytes);
    if (!cdh->uncompressed_file_size()) {
      continue;
    }
    ASSERT_EQ(Z_NO_COMPRESSION, lh->compression_method());
    transient_bytes_->ReadEntryContents(lh);
    ASSERT_EQ(cdh->uncompressed_file_size(), transient_bytes_->data_size());
    struct Sink {
      Sink(const LH *lh)
          : data_start_(lh->data()),
            data_(lh->data()),
            entry_name_(lh->file_name(), lh->file_name_length()) {}
      void operator()(const void *chunk, uint64_t chunk_size) const {
        ASSERT_EQ(0, memcmp(chunk, data_, chunk_size))
            << "Entry " << entry_name_ << "The chunk [" << data_ - data_start_
            << ".." << data_ + chunk_size - data_start_ << ") differs";
        data_ += chunk_size;
      }
      const uint8_t *data_start_;
      mutable const uint8_t *data_;
      std::string entry_name_;
    };
    Sink sink(lh);
    transient_bytes_->stream_out(sink);
  }
  input_jar->Close();
  unlink(kStoredJar);
}

TEST_F(TransientBytesTest, DecompressEntryContents) {
  std::unique_ptr<InputJar> input_jar(new InputJar);
  ASSERT_TRUE(input_jar->Open(kCompressedJar));
  const LH *lh;
  const CDH *cdh;
  std::unique_ptr<Inflater> inflater;
  while ((cdh = input_jar->NextEntry(&lh))) {
    transient_bytes_.reset(new TransientBytes);
    inflater.reset(new Inflater);
    if (!cdh->uncompressed_file_size()) {
      continue;
    }
    ASSERT_EQ(Z_DEFLATED, lh->compression_method());
    transient_bytes_->DecompressEntryContents(cdh, lh, inflater.get());

    ASSERT_EQ(cdh->uncompressed_file_size(), transient_bytes_->data_size());
    // A sink that verifies decompressed entry contents.
    struct Sink {
      Sink(const LH *lh)
          : offset_(0), entry_name_(lh->file_name(), lh->file_name_length()) {}
      void operator()(const void *chunk, uint64_t chunk_size) const {
        for (uint64_t i = 0; i < chunk_size; ++i) {
          // ASSERT_EQ is quite slow in the non-optimized build, avoid calling
          // it 4billion files on a 4GB file.
          if (file_byte_at(offset_ + i) ==
              reinterpret_cast<const uint8_t *>(chunk)[i]) {
            break;
          }
          ASSERT_EQ(file_byte_at(offset_ + i),
                    reinterpret_cast<const uint8_t *>(chunk)[i])
              << "Entry " << entry_name_ << ": mismatch at offset "
              << (offset_ + i);
        }
        offset_ += chunk_size;
      }
      mutable uint64_t offset_;
      std::string entry_name_;
    };
    Sink sink(lh);
    transient_bytes_->stream_out(sink);
  }
  input_jar->Close();
}

// Verify CompressOut: if compressed size is less than original, it writes out
// compressed data.
TEST_F(TransientBytesTest, CompressOut) {
  std::unique_ptr<InputJar> input_jar(new InputJar);
  ASSERT_TRUE(input_jar->Open(kCompressedJar));
  const LH *lh;
  const CDH *cdh;
  std::unique_ptr<Inflater> inflater;
  while ((cdh = input_jar->NextEntry(&lh))) {
    transient_bytes_.reset(new TransientBytes);
    inflater.reset(new Inflater);
    if (!cdh->uncompressed_file_size()) {
      continue;
    }
    ASSERT_EQ(Z_DEFLATED, lh->compression_method());
    transient_bytes_->DecompressEntryContents(cdh, lh, inflater.get());
    ASSERT_EQ(cdh->uncompressed_file_size(), transient_bytes_->data_size());
    // Now let us compress it back.
    uint8_t *buffer =
        reinterpret_cast<uint8_t *>(malloc(cdh->uncompressed_file_size()));
    ASSERT_NE(nullptr, buffer);
    uint32_t crc32 = 0;
    uint64_t bytes_written;
    uint16_t rc = transient_bytes_->CompressOut(buffer, &crc32, &bytes_written);

    EXPECT_EQ(Z_DEFLATED, rc) << "TransientBytes::Write did not compress "
                              << cdh->file_name_string();
    EXPECT_EQ(cdh->crc32(), crc32)
        << "TransientBytes::Write has wrong crc32 for "
        << cdh->file_name_string();

    // Verify contents.
    Inflater inf2;
    inf2.DataToInflate(buffer, 0);  // Just to save the position.
    uint64_t to_inflate = bytes_written;
    uint64_t position = 0;
    while (to_inflate > 0) {
      uint32_t to_inflate_chunk =
          std::min(to_inflate, static_cast<uint64_t>(0xFFFFFFFF));
      inf2.DataToInflate(inf2.next_in(), to_inflate_chunk);
      to_inflate -= to_inflate_chunk;
      for (;;) {
        uint8_t decomp_buf[1024];
        int rc = inf2.Inflate(decomp_buf, sizeof(decomp_buf));
        ASSERT_TRUE(Z_STREAM_END == rc || Z_OK == rc)
            << "Decompressiong contents of " << cdh->file_name_string()
            << " at offset " << position << " returned " << rc;
        for (uint32_t i = 0; i < sizeof(decomp_buf) - inf2.available_out();
             ++i) {
          if (file_byte_at(position) != decomp_buf[i]) {
            EXPECT_EQ(file_byte_at(position), decomp_buf[i])
                << "Decompressed contents of " << cdh->file_name_string()
                << " at offset " << position << " is wrong";
          }
          ++position;
        }
        if (Z_STREAM_END == rc) {
          // Input buffer done.
          break;
        } else {
          EXPECT_EQ(0, inf2.available_out());
        }
      }
    }
    free(buffer);
  }
  input_jar->Close();
}

// Verify CompressOut: if compressed size exceeds original, it writes out
// original data
TEST_F(TransientBytesTest, CompressOutStore) {
  transient_bytes_->Append("a");
  uint8_t buffer[400] = {0xfe, 0xfb};
  uint32_t crc32 = 0;
  uint64_t bytes_written;
  uint16_t rc = transient_bytes_->CompressOut(buffer, &crc32, &bytes_written);
  ASSERT_EQ(Z_NO_COMPRESSION, rc);
  ASSERT_EQ(1, bytes_written);
  ASSERT_EQ('a', buffer[0]);
  ASSERT_EQ(0xfb, buffer[1]);
  ASSERT_EQ(0xE8B7BE43, crc32);
}

// Verify CompressOut: if there are zero bytes in the buffer, just store.
TEST_F(TransientBytesTest, CompressZero) {
  transient_bytes_->Append("");
  uint8_t buffer[400] = {0xfe, 0xfb};
  uint32_t crc32 = 0;
  uint64_t bytes_written;
  uint16_t rc = transient_bytes_->CompressOut(buffer, &crc32, &bytes_written);
  ASSERT_EQ(Z_NO_COMPRESSION, rc);
  ASSERT_EQ(0, bytes_written);
  ASSERT_EQ(0xfe, buffer[0]);
  ASSERT_EQ(0xfb, buffer[1]);
  ASSERT_EQ(0, crc32);
}

// Verify CopyOut.
TEST_F(TransientBytesTest, CopyOut) {
  transient_bytes_->Append("a");
  uint8_t buffer[400] = {0xfe, 0xfb};
  uint32_t crc32 = 0;
  transient_bytes_->CopyOut(buffer, &crc32);
  ASSERT_EQ('a', buffer[0]);
  ASSERT_EQ(0xfb, buffer[1]);
  ASSERT_EQ(0xE8B7BE43, crc32);
}

}  // namespace
