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

#include <memory>

#include "src/tools/singlejar/zlib_interface.h"

#include "gtest/gtest.h"

namespace {

class ZlibInterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inflater_.reset(new Inflater);
    deflater_.reset(new Deflater);
  }

  std::unique_ptr<Inflater> inflater_;
  std::unique_ptr<Deflater> deflater_;
};

static const uint8_t bytes[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

TEST_F(ZlibInterfaceTest, DeflateFully) {
  uint8_t compressed[256];
  deflater_.get()->next_out = compressed;
  deflater_.get()->avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater_->Deflate(bytes, sizeof(bytes), Z_FINISH));
}

TEST_F(ZlibInterfaceTest, DeflateIntoChunks) {
  uint8_t compressed[256];
  deflater_.get()->next_out = compressed;
  deflater_.get()->avail_out = 2;
  EXPECT_EQ(Z_OK, deflater_->Deflate(bytes, sizeof(bytes), Z_FINISH));
  EXPECT_EQ(0, deflater_.get()->avail_out);
  deflater_.get()->next_out = compressed + 2;
  deflater_.get()->avail_out = sizeof(compressed) - 2;
  EXPECT_EQ(Z_STREAM_END,
            deflater_->Deflate(deflater_.get()->next_in,
                               deflater_.get()->avail_in, Z_FINISH));
}

TEST_F(ZlibInterfaceTest, DeflateChunks) {
  uint8_t compressed[256];
  deflater_.get()->next_out = compressed;
  deflater_.get()->avail_out = sizeof(compressed);
  EXPECT_EQ(Z_OK, deflater_->Deflate(bytes, 4, Z_NO_FLUSH));
  EXPECT_EQ(Z_STREAM_END,
            deflater_->Deflate(bytes + 4, sizeof(bytes) - 4, Z_FINISH));
}

TEST_F(ZlibInterfaceTest, InflateFully) {
  uint8_t compressed[256];
  deflater_.get()->next_out = compressed;
  deflater_.get()->avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater_->Deflate(bytes, sizeof(bytes), Z_FINISH));

  // Now we have deflated data, inflate it back and compare.
  size_t compressed_size = sizeof(compressed) - deflater_.get()->avail_out;
  inflater_->DataToInflate(compressed, compressed_size);

  uint8_t uncompressed[256];
  memset(uncompressed, 0, sizeof(uncompressed));
  EXPECT_EQ(Z_STREAM_END,
            inflater_->Inflate(uncompressed, sizeof(uncompressed)));
  EXPECT_EQ(sizeof(bytes), sizeof(uncompressed) - inflater_->available_out());
  EXPECT_EQ(0, memcmp(bytes, uncompressed, sizeof(bytes)));
}

TEST_F(ZlibInterfaceTest, InflateToChunks) {
  uint8_t compressed[256];
  deflater_.get()->next_out = compressed;
  deflater_.get()->avail_out = sizeof(compressed);
  EXPECT_EQ(Z_STREAM_END, deflater_->Deflate(bytes, sizeof(bytes), Z_FINISH));

  // Now we have deflated data, inflate it back and compare.
  size_t compressed_size = sizeof(compressed) - deflater_.get()->avail_out;
  inflater_->DataToInflate(compressed, compressed_size);
  uint8_t uncompressed[256];
  memset(uncompressed, 0, sizeof(uncompressed));
  EXPECT_EQ(Z_OK, inflater_->Inflate(uncompressed, 3));
  EXPECT_EQ(Z_STREAM_END,
            inflater_->Inflate(uncompressed + 3, sizeof(uncompressed) - 3));
  EXPECT_EQ(0, memcmp(bytes, uncompressed, sizeof(bytes)));
}

}  //  namespace
