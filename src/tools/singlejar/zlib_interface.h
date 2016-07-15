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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_ZLIB_INTERFACE_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_ZLIB_INTERFACE_H_

#include <zlib.h>

#include <err.h>
#include <stdint.h>

// An interface to zlib's inflater. Usage:
//   Inflater inflater;
//   inflater.DataToInflate(data, data_size);
//   for (;;) {
//     int rc = inflater.Inflate(out_buffer, out_buffer_size);
//       if (rc == Z_STREAM_END) {
//          break;
//       }
//       // If we ran out of out_buffer, create a new one
//   }
//   inflater.Reset();
//
class Inflater {
 public:
  Inflater() {
    zstream_.zalloc = Z_NULL;
    zstream_.zfree = Z_NULL;
    zstream_.opaque = Z_NULL;
    zstream_.avail_in = 0;
    zstream_.next_in = nullptr;
    int ret = inflateInit2(&zstream_, -MAX_WBITS);
    if (ret != Z_OK) {
      errx(2, "inflateInit2 returned %d\n", ret);
    }
  }

  ~Inflater() { inflateEnd(&zstream_); }

  void reset() { inflateReset(&zstream_); }

  void DataToInflate(const uint8_t *in_buffer, unsigned in_buffer_length) {
    zstream_.next_in = const_cast<uint8_t *>(in_buffer);
    zstream_.avail_in = in_buffer_length;
  }

  int Inflate(uint8_t *out_buffer, unsigned out_buffer_length) {
    zstream_.next_out = out_buffer;
    zstream_.avail_out = out_buffer_length;
    return inflate(&zstream_, Z_SYNC_FLUSH);
  }

  unsigned available_out() const { return zstream_.avail_out; }

  const char *error_message() const { return zstream_.msg; }

 private:
  z_stream zstream_;
};

// A little wrapper around zlib's deflater.
struct Deflater : z_stream {
  Deflater() {
    zalloc = Z_NULL;
    zfree = Z_NULL;
    opaque = Z_NULL;
    next_in = nullptr;
    avail_in = 0;
    next_out = nullptr;
    avail_out = 0;
    int ret = deflateInit2(this, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS,
                           8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
      errx(2, "deflateInit returned %d (%s)", ret, msg);
    }
  }

  ~Deflater() { deflateEnd(this); }

  int Deflate(const uint8_t *data, size_t data_size, int flag) {
    next_in = const_cast<uint8_t *>(data);
    avail_in = data_size;
    return deflate(this, flag);
  }
};

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_ZLIB_INTERFACE_H_
