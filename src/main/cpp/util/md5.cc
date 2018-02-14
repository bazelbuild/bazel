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
/* MD5C.C - RSA Data Security, Inc., MD5 message-digest algorithm
 */

/*
  Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
  rights reserved.

  License to copy and use this software is granted provided that it
  is identified as the "RSA Data Security, Inc. MD5 Message-Digest
  Algorithm" in all material mentioning or referencing this software
  or this function.

  License is also granted to make and use derivative works provided
  that such works are identified as "derived from the RSA Data
  Security, Inc. MD5 Message-Digest Algorithm" in all material
  mentioning or referencing the derived work.

  RSA Data Security, Inc. makes no representations concerning either
  the merchantability of this software or the suitability of this
  software for any particular purpose. It is provided "as is"
  without express or implied warranty of any kind.

  These notices must be retained in any copies of any part of this
  documentation and/or software.
*/

#include "src/main/cpp/util/md5.h"

#include <stddef.h>  // for offsetof
#include <string.h>  // for memcpy

#include <cinttypes>

#if !_STRING_ARCH_unaligned
#if defined(_LP64) || defined(_WIN64)
#  define UNALIGNED_P(p) (reinterpret_cast<uint64_t>(p) % \
                          __alignof__(uint32_t) != 0)  // NOLINT
# else
#  define UNALIGNED_P(p) (reinterpret_cast<uint32_t>(p) % \
                          __alignof__(uint32_t) != 0)  // NOLINT
# endif
#else
#  define UNALIGNED_P(p) (0)
#endif

namespace blaze_util {

using std::string;

static const unsigned int k8Bytes = 64;
static const unsigned int k8ByteMask = 63;

static const unsigned char kPadding[64] = {
  0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// Digit conversion.
static char hex_char[] = "0123456789abcdef";

// This is a templated function so that T can be either a char*
// or a string.  This works because we use the [] operator to access
// individual characters at a time.
template <typename T>
static void b2a_hex_t(const unsigned char* b, T a, int num) {
  for (int i = 0; i < num; i++) {
    a[i * 2 + 0] = hex_char[b[i] >> 4];
    a[i * 2 + 1] = hex_char[b[i] & 0xf];
  }
}

// ----------------------------------------------------------------------
// b2a_hex()
//  Description: Binary-to-Ascii hex conversion.  This converts
//   'num' bytes of binary to a 2*'num'-character hexadecimal representation
//    Return value: 2*'num' characters of ascii text (via the 'to' argument)
// ----------------------------------------------------------------------
static void b2a_hex(const unsigned char* from, string* to, int num) {
  to->resize(num << 1);
  b2a_hex_t<string&>(from, *to, num);
}

Md5Digest::Md5Digest() {
  Reset();
}

Md5Digest::Md5Digest(const Md5Digest& original) {
  memcpy(state, original.state, sizeof(original.state));
  memcpy(count, original.count, sizeof(original.count));
  memcpy(ctx_buffer, original.ctx_buffer, original.ctx_buffer_len);
  ctx_buffer_len = original.ctx_buffer_len;
}

void Md5Digest::Reset() {
  count[0] = count[1] = 0;
  ctx_buffer_len = 0;
  // Load magic initialization constants.
  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;
}

void Md5Digest::Update(const void *buf, unsigned int length) {
  const unsigned char *input = reinterpret_cast<const unsigned char*>(buf);
  unsigned int buffer_space_len;

  buffer_space_len = k8Bytes - ctx_buffer_len;

  // Transform as many times as possible.
  if (length >= buffer_space_len) {
    if (buffer_space_len != 0 && ctx_buffer_len != 0) {
      // Copy more bytes to fill the complete buffer
      memcpy(ctx_buffer + ctx_buffer_len, input, buffer_space_len);
      Transform(ctx_buffer, k8Bytes);
      input += buffer_space_len;
      length -= buffer_space_len;
      ctx_buffer_len = 0;
    }

    if (UNALIGNED_P(input)) {
      while (length >= k8Bytes) {
        memcpy(ctx_buffer, input, k8Bytes);
        Transform(ctx_buffer, k8Bytes);
        input += k8Bytes;
        length -= k8Bytes;
      }
    } else if (length >= k8Bytes) {
      Transform(input, length & ~k8ByteMask);
      input += length & ~k8ByteMask;
      length &= k8ByteMask;
    }
  }

  // Buffer remaining input
  memcpy(ctx_buffer + ctx_buffer_len, input, length);
  ctx_buffer_len += length;
}

void Md5Digest::Finish(unsigned char digest[16]) {
  count[0] += ctx_buffer_len;
  if (count[0] < ctx_buffer_len) {
    ++count[1];
  }

  /* Put the 64-bit file length in *bits* at the end of the buffer.  */
  unsigned int size = (ctx_buffer_len < 56 ? 64 : 128);
  uint32_t words[2] = { count[0] << 3, (count[1] << 3) | (count[0] >> 29) };
  memcpy(ctx_buffer + size - 8, words, 8);

  memcpy(ctx_buffer + ctx_buffer_len, kPadding, size - 8 - ctx_buffer_len);

  Transform(ctx_buffer, size);

  uint32_t* r = reinterpret_cast<uint32_t*>(digest);
  r[0] = state[0];
  r[1] = state[1];
  r[2] = state[2];
  r[3] = state[3];
}

void Md5Digest::Transform(
    const unsigned char* buffer, unsigned int len) {
  // Constants for transform routine.
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

  // F, G, H and I are basic MD5 functions.
/* These are the four functions used in the four steps of the MD5 algorithm
   and defined in the RFC 1321.  The first function is a little bit optimized
   (as found in Colin Plumbs public domain implementation).  */
/* #define F(b, c, d) ((b & c) | (~b & d)) */
#define F(x, y, z) (z ^ (x & (y ^ z)))
#define G(x, y, z) F (z, x, y)
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

  // ROTATE_LEFT rotates x left n bits.
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

  // FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
  // Rotation is separate from addition to prevent recomputation.
#define FF(a, b, c, d, s, ac) { \
      (a) += F((b), (c), (d)) + ((*x_pos++ = *cur_word++)) + \
          static_cast<uint32_t>(ac); \
      (a) = ROTATE_LEFT((a), (s)); \
      (a) += (b); \
    }

#define GG(a, b, c, d, x, s, ac) { \
      (a) += G((b), (c), (d)) + (x) + static_cast<uint32_t>(ac); \
      (a) = ROTATE_LEFT((a), (s)); \
      (a) += (b); \
     }
#define HH(a, b, c, d, x, s, ac) { \
      (a) += H((b), (c), (d)) + (x) + static_cast<uint32_t>(ac); \
      (a) = ROTATE_LEFT((a), (s)); \
      (a) += (b); \
     }
#define II(a, b, c, d, x, s, ac) { \
      (a) += I((b), (c), (d)) + (x) + static_cast<uint32_t>(ac); \
      (a) = ROTATE_LEFT((a), (s)); \
      (a) += (b); \
     }

  count[0] += len;
  if (count[0] < len) {
    ++count[1];
  }

  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t x[16];

  const uint32_t *cur_word = reinterpret_cast<const uint32_t*>(buffer);
  const uint32_t *end_word = cur_word + (len / sizeof(uint32_t));

  while (cur_word < end_word) {
    uint32_t *x_pos = x;
    uint32_t prev_a = a;
    uint32_t prev_b = b;
    uint32_t prev_c = c;
    uint32_t prev_d = d;

    // Round 1
    FF(a, b, c, d, S11, 0xd76aa478);  // 1
    FF(d, a, b, c, S12, 0xe8c7b756);  // 2
    FF(c, d, a, b, S13, 0x242070db);  // 3
    FF(b, c, d, a, S14, 0xc1bdceee);  // 4
    FF(a, b, c, d, S11, 0xf57c0faf);  // 5
    FF(d, a, b, c, S12, 0x4787c62a);  // 6
    FF(c, d, a, b, S13, 0xa8304613);  // 7
    FF(b, c, d, a, S14, 0xfd469501);  // 8
    FF(a, b, c, d, S11, 0x698098d8);  // 9
    FF(d, a, b, c, S12, 0x8b44f7af);  // 10
    FF(c, d, a, b, S13, 0xffff5bb1);  // 11
    FF(b, c, d, a, S14, 0x895cd7be);  // 12
    FF(a, b, c, d, S11, 0x6b901122);  // 13
    FF(d, a, b, c, S12, 0xfd987193);  // 14
    FF(c, d, a, b, S13, 0xa679438e);  // 15
    FF(b, c, d, a, S14, 0x49b40821);  // 16

    // Round 2
    GG(a, b, c, d, x[ 1], S21, 0xf61e2562);  // 17
    GG(d, a, b, c, x[ 6], S22, 0xc040b340);  // 18
    GG(c, d, a, b, x[11], S23, 0x265e5a51);  // 19
    GG(b, c, d, a, x[ 0], S24, 0xe9b6c7aa);  // 20
    GG(a, b, c, d, x[ 5], S21, 0xd62f105d);  // 21
    GG(d, a, b, c, x[10], S22,  0x2441453);  // 22
    GG(c, d, a, b, x[15], S23, 0xd8a1e681);  // 23
    GG(b, c, d, a, x[ 4], S24, 0xe7d3fbc8);  // 24
    GG(a, b, c, d, x[ 9], S21, 0x21e1cde6);  // 25
    GG(d, a, b, c, x[14], S22, 0xc33707d6);  // 26
    GG(c, d, a, b, x[ 3], S23, 0xf4d50d87);  // 27
    GG(b, c, d, a, x[ 8], S24, 0x455a14ed);  // 28
    GG(a, b, c, d, x[13], S21, 0xa9e3e905);  // 29
    GG(d, a, b, c, x[ 2], S22, 0xfcefa3f8);  // 30
    GG(c, d, a, b, x[ 7], S23, 0x676f02d9);  // 31
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);  // 32

    // Round 3
    HH(a, b, c, d, x[ 5], S31, 0xfffa3942);  // 33
    HH(d, a, b, c, x[ 8], S32, 0x8771f681);  // 34
    HH(c, d, a, b, x[11], S33, 0x6d9d6122);  // 35
    HH(b, c, d, a, x[14], S34, 0xfde5380c);  // 36
    HH(a, b, c, d, x[ 1], S31, 0xa4beea44);  // 37
    HH(d, a, b, c, x[ 4], S32, 0x4bdecfa9);  // 38
    HH(c, d, a, b, x[ 7], S33, 0xf6bb4b60);  // 39
    HH(b, c, d, a, x[10], S34, 0xbebfbc70);  // 40
    HH(a, b, c, d, x[13], S31, 0x289b7ec6);  // 41
    HH(d, a, b, c, x[ 0], S32, 0xeaa127fa);  // 42
    HH(c, d, a, b, x[ 3], S33, 0xd4ef3085);  // 43
    HH(b, c, d, a, x[ 6], S34,  0x4881d05);  // 44
    HH(a, b, c, d, x[ 9], S31, 0xd9d4d039);  // 45
    HH(d, a, b, c, x[12], S32, 0xe6db99e5);  // 46
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8);  // 47
    HH(b, c, d, a, x[ 2], S34, 0xc4ac5665);  // 48

    // Round 4
    II(a, b, c, d, x[ 0], S41, 0xf4292244);  // 49
    II(d, a, b, c, x[ 7], S42, 0x432aff97);  // 50
    II(c, d, a, b, x[14], S43, 0xab9423a7);  // 51
    II(b, c, d, a, x[ 5], S44, 0xfc93a039);  // 52
    II(a, b, c, d, x[12], S41, 0x655b59c3);  // 53
    II(d, a, b, c, x[ 3], S42, 0x8f0ccc92);  // 54
    II(c, d, a, b, x[10], S43, 0xffeff47d);  // 55
    II(b, c, d, a, x[ 1], S44, 0x85845dd1);  // 56
    II(a, b, c, d, x[ 8], S41, 0x6fa87e4f);  // 57
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0);  // 58
    II(c, d, a, b, x[ 6], S43, 0xa3014314);  // 59
    II(b, c, d, a, x[13], S44, 0x4e0811a1);  // 60
    II(a, b, c, d, x[ 4], S41, 0xf7537e82);  // 61
    II(d, a, b, c, x[11], S42, 0xbd3af235);  // 62
    II(c, d, a, b, x[ 2], S43, 0x2ad7d2bb);  // 63
    II(b, c, d, a, x[ 9], S44, 0xeb86d391);  // 64

    a += prev_a;
    b += prev_b;
    c += prev_c;
    d += prev_d;
  }

  state[0] = a;
  state[1] = b;
  state[2] = c;
  state[3] = d;
}

string Md5Digest::String() const {
  string result;
  b2a_hex(reinterpret_cast<const uint8_t*>(state), &result, 16);
  return result;
}

}  // namespace blaze_util
