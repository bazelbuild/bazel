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

#include "src/tools/singlejar/zlib_interface.h"

#include "googletest/include/gtest/gtest.h"

namespace {

static const uint8_t bytes[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

TEST(ZlibInterfaceTest, DeflateFully) {
  Deflater deflater;
  uint8_t compressed[256];
  deflater.next_out = compressed;
  deflater.avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater.Deflate(bytes, sizeof(bytes), Z_FINISH));
}

TEST(ZlibInterfaceTest, DeflateIntoChunks) {
  Deflater deflater;
  uint8_t compressed[256];
  deflater.next_out = compressed;
  deflater.avail_out = 2;
  EXPECT_EQ(Z_OK, deflater.Deflate(bytes, sizeof(bytes), Z_FINISH));
  EXPECT_EQ(0UL, deflater.avail_out);
  deflater.next_out = compressed + 2;
  deflater.avail_out = sizeof(compressed) - 2;
  EXPECT_EQ(Z_STREAM_END,
            deflater.Deflate(deflater.next_in,
                               deflater.avail_in, Z_FINISH));
}

TEST(ZlibInterfaceTest, DeflateChunks) {
  Deflater deflater;
  uint8_t compressed[256];
  deflater.next_out = compressed;
  deflater.avail_out = sizeof(compressed);
  EXPECT_EQ(Z_OK, deflater.Deflate(bytes, 4, Z_NO_FLUSH));
  EXPECT_EQ(Z_STREAM_END,
            deflater.Deflate(bytes + 4, sizeof(bytes) - 4, Z_FINISH));
}

TEST(ZlibInterfaceTest, InflateFully) {
  uint8_t compressed[256];
  Deflater deflater;
  deflater.next_out = compressed;
  deflater.avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater.Deflate(bytes, sizeof(bytes), Z_FINISH));

  // Now we have deflated data, inflate it back and compare.
  size_t compressed_size = sizeof(compressed) - deflater.avail_out;
  Inflater inflater;
  inflater.DataToInflate(compressed, compressed_size);

  uint8_t uncompressed[256];
  memset(uncompressed, 0, sizeof(uncompressed));
  EXPECT_EQ(Z_STREAM_END,
            inflater.Inflate(uncompressed, sizeof(uncompressed)));
  EXPECT_EQ(sizeof(bytes), sizeof(uncompressed) - inflater.available_out());
  EXPECT_EQ(0, memcmp(bytes, uncompressed, sizeof(bytes)));
}

TEST(ZlibInterfaceTest, InflateToChunks) {
  uint8_t compressed[256];
  Deflater deflater;
  deflater.next_out = compressed;
  deflater.avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater.Deflate(bytes, sizeof(bytes), Z_FINISH));

  // Now we have deflated data, inflate it back and compare.
  size_t compressed_size = sizeof(compressed) - deflater.avail_out;
  Inflater inflater;
  inflater.DataToInflate(compressed, compressed_size);
  uint8_t uncompressed[256];
  memset(uncompressed, 0, sizeof(uncompressed));
  EXPECT_EQ(Z_OK, inflater.Inflate(uncompressed, 3));
  EXPECT_EQ(Z_STREAM_END,
            inflater.Inflate(uncompressed + 3, sizeof(uncompressed) - 3));
  EXPECT_EQ(0, memcmp(bytes, uncompressed, sizeof(bytes)));
}

}  //  namespace
