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

#include <stdlib.h>
#include <algorithm>
#include <cstdio>

#include "third_party/ijar/common.h"
#include "third_party/ijar/zlib_client.h"
#include <zlib.h>

namespace devtools_ijar {

u4 ComputeCrcChecksum(u1 *buf, size_t length) {
  return crc32(0, buf, length);
}

size_t TryDeflate(u1 *buf, size_t length) {
  u1 *outbuf = reinterpret_cast<u1 *>(malloc(length));
  z_stream stream;

  // Initialize the z_stream strcut for reading from buf and wrinting in outbuf.
  stream.zalloc = Z_NULL;
  stream.zfree = Z_NULL;
  stream.opaque = Z_NULL;
  stream.total_in = length;
  stream.avail_in = length;
  stream.total_out = length;
  stream.avail_out = length;
  stream.next_in = buf;
  stream.next_out = outbuf;

  // deflateInit2 negative windows size prevent the zlib wrapper to be used.
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK) {
    // Failure to compress => return the buffer uncompressed
    free(outbuf);
    return length;
  }

  if (deflate(&stream, Z_FINISH) == Z_STREAM_END) {
    // Compression successful and fits in outbuf, let's copy the result in buf.
    length = stream.total_out;
    memcpy(buf, outbuf, length);
  }

  deflateEnd(&stream);
  free(outbuf);

  // Return the length of the resulting buffer
  return length;
}

Decompressor::Decompressor() {
  uncompressed_data_allocated_ = INITIAL_BUFFER_SIZE;
  uncompressed_data_ =
      reinterpret_cast<u1 *>(malloc(uncompressed_data_allocated_));
}

Decompressor::~Decompressor() { free(uncompressed_data_); }

DecompressedFile *Decompressor::UncompressFile(const u1 *buffer,
                                               size_t bytes_avail) {
  z_stream stream;

  stream.zalloc = Z_NULL;
  stream.zfree = Z_NULL;
  stream.opaque = Z_NULL;
  stream.avail_in = bytes_avail;
  stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(buffer));

  int ret = inflateInit2(&stream, -MAX_WBITS);
  if (ret != Z_OK) {
    error("inflateInit: %d\n", ret);
    return NULL;
  }

  int uncompressed_until_now = 0;

  while (true) {
    stream.avail_out = uncompressed_data_allocated_ - uncompressed_until_now;
    stream.next_out = uncompressed_data_ + uncompressed_until_now;
    int old_avail_out = stream.avail_out;

    ret = inflate(&stream, Z_SYNC_FLUSH);
    int uncompressed_now = old_avail_out - stream.avail_out;
    uncompressed_until_now += uncompressed_now;

    switch (ret) {
      case Z_STREAM_END: {
        struct DecompressedFile *decompressedFile =
            reinterpret_cast<DecompressedFile *>(
                malloc(sizeof(DecompressedFile)));
        // zlib said that there is no more data to decompress.
        u1 *new_p = reinterpret_cast<u1 *>(stream.next_in);
        decompressedFile->compressed_size = new_p - buffer;
        decompressedFile->uncompressed_size = uncompressed_until_now;
        decompressedFile->uncompressed_data = uncompressed_data_;
        inflateEnd(&stream);
        return decompressedFile;
      }

      case Z_OK: {
        // zlib said that there is no more room in the buffer allocated for
        // the decompressed data. Enlarge that buffer and try again.

        if (uncompressed_data_allocated_ == MAX_BUFFER_SIZE) {
          error(
              "ijar does not support decompressing files "
              "larger than %dMB.\n",
              static_cast<int>((MAX_BUFFER_SIZE / (1024 * 1024))));
          return NULL;
        }

        uncompressed_data_allocated_ *= 2;
        if (uncompressed_data_allocated_ > MAX_BUFFER_SIZE) {
          uncompressed_data_allocated_ = MAX_BUFFER_SIZE;
        }

        uncompressed_data_ = reinterpret_cast<u1 *>(
            realloc(uncompressed_data_, uncompressed_data_allocated_));
        break;
      }

      case Z_DATA_ERROR:
      case Z_BUF_ERROR:
      case Z_STREAM_ERROR:
      case Z_NEED_DICT:
      default: {
        error("zlib returned error code %d during inflate.\n", ret);
        return NULL;
      }
    }
  }
}

char *Decompressor::GetError() {
  if (errmsg[0] == 0) {
    return NULL;
  }
  return errmsg;
}

int Decompressor::error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(errmsg, 4 * PATH_MAX, fmt, ap);
  va_end(ap);
  return -1;
}
}  // namespace devtools_ijar
