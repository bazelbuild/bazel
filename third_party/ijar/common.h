// Copyright 2015 The Bazel Authors. All rights reserved.
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
//
// common.h -- common definitions.
//

#ifndef INCLUDED_DEVTOOLS_IJAR_COMMON_H
#define INCLUDED_DEVTOOLS_IJAR_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#define PATH_MAX 4096
typedef int mode_t;
#endif  // _WIN32

namespace devtools_ijar {

typedef unsigned long long u8;
typedef uint32_t u4;
typedef uint16_t u2;
typedef uint8_t  u1;

// be = big endian, le = little endian

inline u1 get_u1(const u1 *&p) {
    return *p++;
}

inline u2 get_u2be(const u1 *&p) {
    u4 x = (p[0] << 8) | p[1];
    p += 2;
    return x;
}

inline u2 get_u2le(const u1 *&p) {
    u4 x = (p[1] << 8) | p[0];
    p += 2;
    return x;
}

inline u4 get_u4be(const u1 *&p) {
    u4 x = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
    p += 4;
    return x;
}

inline u4 get_u4le(const u1 *&p) {
    u4 x = (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
    p += 4;
    return x;
}

inline u8 get_u8le(const u1 *&p) {
  u4 lo = get_u4le(p);
  u4 hi = get_u4le(p);
  u8 x = ((u8)hi << 32) | lo;
  return x;
}

inline void put_u1(u1 *&p, u1 x) {
    *p++ = x;
}

inline void put_u2be(u1 *&p, u2 x) {
    *p++ = x >> 8;
    *p++ = x & 0xff;
}

inline void put_u2le(u1 *&p, u2 x) {
    *p++ = x & 0xff;
    *p++ = x >> 8;;
}

inline void put_u4be(u1 *&p, u4 x) {
    *p++ = x >> 24;
    *p++ = (x >> 16) & 0xff;
    *p++ = (x >> 8) & 0xff;
    *p++ = x & 0xff;
}

inline void put_u4le(u1 *&p, u4 x) {
    *p++ = x & 0xff;
    *p++ = (x >> 8) & 0xff;
    *p++ = (x >> 16) & 0xff;
    *p++ = x >> 24;
}

inline void put_u8le(u1 *&p, u8 x) {
  put_u4le(p, x & 0xffffffff);
  put_u4le(p, (x >> 32) & 0xffffffff);
}

// Copy n bytes from src to p, and advance p.
inline void put_n(u1 *&p, const u1 *src, size_t n) {
  memcpy(p, src, n);
  p += n;
}

struct Reader {
  const u1 *p;
  const u1 *end;
  bool ok;

  Reader(const u1 *p, size_t len) : p(p), end(p + len), ok(true) {}
  Reader(const u1 *p, const u1 *end) : p(p), end(end), ok(p <= end) {}

  size_t remaining() const { return ok ? static_cast<size_t>(end - p) : 0; }

  bool Ensure(size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    return true;
  }

  u1 get_u1() { return Ensure(1) ? devtools_ijar::get_u1(p) : 0; }
  u2 get_u2be() { return Ensure(2) ? devtools_ijar::get_u2be(p) : 0; }
  u2 get_u2le() { return Ensure(2) ? devtools_ijar::get_u2le(p) : 0; }
  u4 get_u4be() { return Ensure(4) ? devtools_ijar::get_u4be(p) : 0; }
  u4 get_u4le() { return Ensure(4) ? devtools_ijar::get_u4le(p) : 0; }
  u8 get_u8le() { return Ensure(8) ? devtools_ijar::get_u8le(p) : 0; }

  const u1 *get_bytes(size_t n) {
    if (!Ensure(n)) return nullptr;
    const u1 *res = p;
    p += n;
    return res;
  }

  Reader slice(size_t n) {
    if (!Ensure(n)) {
      Reader bad(end, end);
      bad.ok = false;
      return bad;
    }
    const u1 *start = p;
    p += n;
    return Reader(start, start + n);
  }
};

// Reads a JVM class from classdata_in (of the specified length), and
// writes out a simplified class to classdata_out, advancing the
// pointer. Returns true if the class should be kept.
bool StripClass(u1 *&classdata_out, const u1 *classdata_in, size_t in_length);

extern bool verbose;

}  // namespace devtools_ijar

#endif // INCLUDED_DEVTOOLS_IJAR_COMMON_H
