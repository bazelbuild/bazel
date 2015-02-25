// Copyright 2001,2007 Alan Donovan. All rights reserved.
//
// Author: Alan Donovan <adonovan@google.com>
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

// Copy n bytes from src to p, and advance p.
inline void put_n(u1 *&p, const u1 *src, size_t n) {
  memcpy(p, src, n);
  p += n;
}


// Opens "file_in" (a .jar file) for reading, and writes an interface
// .jar to "file_out".  Returns zero on success.
int OpenFilesAndProcessJar(const char *file_out, const char *file_in);


// Reads a JVM class from classdata_in (of the specified length), and
// writes out a simplified class to classdata_out, advancing the
// pointer.
void StripClass(u1 *&classdata_out, const u1 *classdata_in, size_t in_length);

extern bool verbose;

// Given the data in the zip file, returns the offset of the central
// directory and the number of files contained in it in *offset and
// *entries, respectively.  Returns true on success, or false on error.
bool FindZipCentralDirectory(const u1* bytes, size_t in_length, u4* offset);

}  // namespace devtools_ijar

#endif // INCLUDED_DEVTOOLS_IJAR_COMMON_H
