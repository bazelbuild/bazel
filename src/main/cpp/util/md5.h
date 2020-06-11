// Copyright 2014 The Bazel Authors. All rights reserved.
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
// Provides a fast MD5 implementation.
//
// This implementation saves us from linking huge OpenSSL library.

#ifndef BAZEL_SRC_MAIN_CPP_UTIL_MD5_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_MD5_H_

#include <string>

#if defined(_MSC_VER) && !defined(__alignof__)
#define __alignof__ __alignof
#endif  // _MSC_VER && !__alignof__

namespace blaze_util {

// The <code>Context</code> class performs the actual MD5
// computation. It works incrementally and can be fed a single byte at
// a time if desired.
class Md5Digest {
 public:
  Md5Digest();

  Md5Digest(const Md5Digest& original);

  // the MD5 digest is always 128 bits = 16 bytes
  static constexpr int kDigestLength = 16;

  // Resets the context so that it can be used to calculate another
  // MD5 digest. The context is in the same state as if it had just
  // been constructed. It is unnecessary to call <code>Reset</code> on
  // a newly created context.
  void Reset();

  // Add <code>length</code> bytes of <code>buf</code> to the MD5
  // digest.
  void Update(const void *buf, unsigned int length);

  // Retrieve the computed MD5 digest as a 16 byte array.
  void Finish(unsigned char* digest);

  // Produces a hexadecimal string representation of this digest in the form:
  // [0-9a-f]{32}
  std::string String() const;

 private:
  void Transform(const unsigned char* buffer, unsigned int len);

 private:
  unsigned int state[4];          // state (ABCD)
  unsigned int count[2];          // number of bits, modulo 2^64 (lsb first)
  unsigned char ctx_buffer[128];  // input buffer
  unsigned int ctx_buffer_len;
};

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_MD5_H_
