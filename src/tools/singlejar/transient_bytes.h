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
#ifndef SRC_TOOLS_SINGLEJAR_TRANSIENT_BYTES_H_
#define SRC_TOOLS_SINGLEJAR_TRANSIENT_BYTES_H_

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#include <inttypes.h>
#include <algorithm>
#include <ostream>

#include "src/tools/singlejar/diag.h"
#include "src/tools/singlejar/zip_headers.h"
#include "src/tools/singlejar/zlib_interface.h"

/*
 * An instance of this class holds decompressed data in a list of chunks,
 * to be eventually compressed to the output buffer.
 * Use DecompressFile() or ReadFile() (depending on whether an entry is
 * compressed or not) to append the contents of a Zip entry.
 * Use Append() to append a sequence of bytes or a string.
 * Use Write() to write out the contents, it will compress the entry if
 * necessary.
 */
class TransientBytes {
 public:
  TransientBytes()
      : allocated_(0),
        data_size_(0),
        first_block_(nullptr),
        last_block_(nullptr) {}

  ~TransientBytes() {
    while (first_block_) {
      auto block = first_block_;
      first_block_ = first_block_->next_block_;
      delete block;
    }
    last_block_ = nullptr;
  }

  // Appends raw bytes.
  void Append(const uint8_t *data, uint64_t data_size) {
    uint64_t chunk_size;
    auto data_end = data + data_size;
    for (; data < data_end; data += chunk_size) {
      chunk_size =
          std::min(static_cast<uint64_t>(data_end - data), ensure_space());
      copy(data, chunk_size);
    }
  }

  // Same, but for a string.
  void Append(const char *str) {
    Append(reinterpret_cast<const uint8_t *>(str), strlen(str));
  }

  // Appends the contents of the uncompressed Zip entry.
  void ReadEntryContents(const LH *lh) {
    Append(lh->data(), lh->uncompressed_file_size());
  }

  // Appends the contents of the compressed Zip entry. Resets the inflater
  // used to decompress.
  void DecompressEntryContents(const CDH *cdh, const LH *lh,
                               Inflater *inflater) {
    uint64_t old_total_out = inflater->total_out();
    uint64_t in_bytes;
    uint64_t out_bytes;
    const uint8_t *data = lh->data();

    if (cdh->no_size_in_local_header()) {
      in_bytes = cdh->compressed_file_size();
      out_bytes = cdh->uncompressed_file_size();
    } else {
      in_bytes = lh->compressed_file_size();
      out_bytes = lh->uncompressed_file_size();
    }

    while (in_bytes > 0) {
      // A single region to inflate cannot exceed 4GB-1.
      uint32_t in_bytes_chunk = 0xFFFFFFFF;
      if (in_bytes_chunk > in_bytes) {
        in_bytes_chunk = in_bytes;
      }
      inflater->DataToInflate(data, in_bytes_chunk);
      for (;;) {
        uint32_t available_out = ensure_space();
        int ret = inflater->Inflate(append_position(), available_out);
        uint32_t inflated = available_out - inflater->available_out();
        if (Z_STREAM_END == ret) {
          // No more data to decompress. Update write position and we are done
          // for this input chunk.
          advance(inflated);
          break;
        } else if (Z_OK == ret) {
          // No more space in the output buffer. Advance write position, update
          // the number of remaining bytes.
          if (inflater->available_out()) {
            diag_errx(2,
                      "%s:%d: Internal error inflating %.*s: Inflate reported "
                      "Z_OK but there are still %" PRIu32
                      " bytes available in the output buffer",
                      __FILE__, __LINE__, lh->file_name_length(),
                      lh->file_name(), inflater->available_out());
          }
          advance(inflated);
        } else {
          diag_errx(2,
                    "%s:%d: Internal error inflating %.*s: inflate() call "
                    "returned %d (%s)",
                    __FILE__, __LINE__, lh->file_name_length(), lh->file_name(),
                    ret, inflater->error_message());
        }
      }
      data += in_bytes_chunk;
      in_bytes -= in_bytes_chunk;
    }

    // Smog check
    // This check is disabled on Windows because z_stream::total_out is of type
    // of uLong (unsigned long), which is 64-bit for most 64-bit Unix platforms,
    // but it is 32-bit even for Win64. This means even though zlib is capable
    // of compressing data >4GB as long as it is processed by chunks, zlib
    // cannot report the correct total number of processed bytes >4GB through
    // z_stream::total_out on Windows.
#ifndef _WIN32
    if (inflater->total_out() - old_total_out != out_bytes) {
      diag_errx(2,
                "%s:%d: Internal error inflating %.*s: inflater wrote %" PRIu64
                " bytes , but the uncompressed entry should be %" PRIu64
                "bytes long",
                __FILE__, __LINE__, lh->file_name_length(), lh->file_name(),
                inflater->total_out() - old_total_out, out_bytes);
    }
#endif
    inflater->reset();
  }

  // Writes the contents bytes to the given buffer in an optimal way, i.e., the
  // shorter of compressed or uncompressed. Sets the checksum and number of
  // bytes written and returns Z_DEFLATED if compression took place or
  // Z_NO_COMPRESSION otherwise.
  uint16_t CompressOut(uint8_t *buffer, uint32_t *checksum,
                       uint64_t *bytes_written) {
    *checksum = 0;
    uint64_t to_compress = data_size();
    if (to_compress == 0) {
      *bytes_written = 0;
      return Z_NO_COMPRESSION;
    }

    Deflater deflater;
    deflater.next_out = buffer;
    uint16_t compression_method = Z_DEFLATED;

    // Feed data blocks to the deflater one by one, but break if the compressed
    // size exceeds the original size.
    for (auto data_block = first_block_;
         data_block && compression_method != Z_NO_COMPRESSION;
         data_block = data_block->next_block_) {
      // The compressed size should not exceed the original size less the number
      // of bytes already compressed. And, it should not exceed 4GB-1.
      deflater.avail_out = std::min(data_size() - deflater.total_out,
                                    static_cast<uint64_t>(0xFFFFFFFF));
      // Out of the total number of bytes that remain to be compressed, we
      // can compress no more than this block.
      uint32_t chunk_size = static_cast<uint32_t>(std::min(
          static_cast<uint64_t>(sizeof(data_block->data_)), to_compress));
      *checksum = crc32(*checksum, data_block->data_, chunk_size);
      deflater.avail_in = chunk_size;
      to_compress -= chunk_size;
      int ret = deflater.Deflate(data_block->data_, chunk_size,
                                 to_compress ? Z_NO_FLUSH : Z_FINISH);
      if (ret == Z_OK) {
        if (!deflater.avail_out) {
          // We ran out of space in the output buffer, which means
          // that deflated size exceeds original size. Leave the loop
          // and just copy the data.
          compression_method = Z_NO_COMPRESSION;
        }
      } else if (ret == Z_BUF_ERROR && !deflater.avail_in) {
        // We ran out of data block, this is not a error.
      } else if (ret == Z_STREAM_END) {
        if (data_block->next_block_ || to_compress) {
          diag_errx(2,
                    "%s:%d: Internal error: deflate() call at the end, but "
                    "there is more data to compress!",
                    __FILE__, __LINE__);
        }
      } else {
        diag_errx(2, "%s:%d: deflate error %d(%s)", __FILE__, __LINE__, ret,
                  deflater.msg);
      }
    }
    if (compression_method != Z_NO_COMPRESSION) {
      *bytes_written = deflater.total_out;
      return compression_method;
    }

    // Compression does not help, just copy the bytes to the output buffer.
    CopyOut(buffer, checksum);
    *bytes_written = data_size();
    return Z_NO_COMPRESSION;
  }

  // Copies the bytes to the buffer and sets the checksum.
  void CopyOut(uint8_t *buffer, uint32_t *checksum) {
    uint64_t to_copy = data_size();
    uint8_t *buffer_end = buffer + to_copy;
    *checksum = 0;
    for (auto data_block = first_block_; data_block;
         data_block = data_block->next_block_) {
      size_t chunk_size =
          std::min(static_cast<uint64_t>(sizeof(data_block->data_)), to_copy);
      *checksum = crc32(*checksum, data_block->data_, chunk_size);
      memcpy(buffer_end - to_copy, data_block->data_, chunk_size);
      to_copy -= chunk_size;
    }
  }

  // Number of data bytes.
  uint64_t data_size() const { return data_size_; }

  // This is mostly for testing: stream out contents to a Sink instance.
  // The class Sink has to have
  //     void operator()(const void *chunk, uint64_t chunk_size) const;
  //
  template <class Sink>
  void stream_out(const Sink &sink) const {
    uint64_t to_copy = data_size();
    for (auto data_block = first_block_; data_block;
         data_block = data_block->next_block_) {
      uint64_t chunk_size = sizeof(data_block->data_);
      if (chunk_size > to_copy) {
        chunk_size = to_copy;
      }
      sink.operator()(data_block->data_, chunk_size);
      to_copy -= chunk_size;
    }
  }

  uint8_t last_byte() const {
    if (!data_size()) {
      diag_errx(1, "%s:%d: last_char() cannot be called if buffer is empty",
                __FILE__, __LINE__);
    }
    if (free_size() >= sizeof(last_block_->data_)) {
      diag_errx(1, "%s:%d: internal error: the last data block is empty",
                __FILE__, __LINE__);
    }
    return *(last_block_->End() - free_size() - 1);
  }

 private:
  // Ensures there is some space to write to, returns the amount available.
  uint64_t ensure_space() {
    if (!free_size()) {
      auto *data_block = new DataBlock();
      if (last_block_) {
        last_block_->next_block_ = data_block;
      }
      last_block_ = data_block;
      if (!first_block_) {
        first_block_ = data_block;
      }
      allocated_ += sizeof(data_block->data_);
    }
    return free_size();
  }

  // Records that given amount of bytes is to be appended to the buffer.
  // Returns the old write position.
  uint8_t *advance(size_t amount) {
    if (amount > free_size()) {
      diag_errx(
          2, "%s: %d: Cannot advance %zu bytes, only %" PRIu64 " is available",
          __FILE__, __LINE__, amount, free_size());
    }
    uint8_t *pos = append_position();
    data_size_ += amount;
    return pos;
  }

  void copy(const uint8_t *from, size_t count) {
    memcpy(advance(count), from, count);
  }

  uint8_t *append_position() {
    return last_block_ ? last_block_->End() - free_size() : nullptr;
  }

  // Returns the amount of free space.
  uint64_t free_size() const { return allocated_ - data_size_; }

  // The bytes are kept in an linked list of the DataBlock instances.
  // TODO(asmundak): perhaps use mmap to allocate these?
  struct DataBlock {
    struct DataBlock *next_block_;
    uint8_t data_[0x40000 - 8];
    DataBlock() : next_block_(nullptr) {}
    uint8_t *End() { return data_ + sizeof(data_); }
  };

  uint64_t allocated_;
  uint64_t data_size_;
  struct DataBlock *first_block_;
  struct DataBlock *last_block_;
};

#endif  // SRC_TOOLS_SINGLEJAR_TRANSIENT_BYTES_H_
