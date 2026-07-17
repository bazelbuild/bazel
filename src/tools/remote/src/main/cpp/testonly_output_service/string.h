// Copyright 2024 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_STRING_H_
#define BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_STRING_H_

#include <stdint.h>
#include <string.h>

#include "src/tools/remote/src/main/cpp/testonly_output_service/memory.h"

// Null terminated utf-8 string.
struct Str8 {
  uint8_t *ptr;
  // Length in bytes, excluding the null terminator. However, the string is
  // always null terminated which means the memory pointed to by ptr must be
  // at least len + 1 bytes long.
  size_t len;
};

// Returns true if the string is empty.
static inline bool IsEmptyStr8(Str8 str) {
  bool result = str.len == 0;
  return result;
}

// Constructs a Str8 from a C string.
static inline Str8 Str8FromCStr(const char *str) {
  Str8 result = {(uint8_t *)str, strlen(str)};
  return result;
}

// Returns true if the string starts with the given prefix.
static inline bool StartsWithStr8(Str8 str, Str8 prefix) {
  bool result =
      str.len >= prefix.len && memcmp(str.ptr, prefix.ptr, prefix.len) == 0;
  return result;
}

// The result of parsing an unsigned 32-bit integer from a string. The `value`
// is only valid if `valid` is true.
struct ParsedUInt32 {
  bool valid;
  uint32_t value;
};

// Parses an unsigned 32-bit integer from the given string.
ParsedUInt32 ParseUInt32(Str8 str);

// Pushes a copy of the given string to the arena
Str8 PushStr8(Arena *arena, Str8 str);
// Pushes a formatted string to the arena.
Str8 PushStr8F(Arena *arena, const char *format, ...);
// Pushes a substring of the given string to the arena, from the index `begin`
// to the end of the string.
Str8 PushSubStr8(Arena *arena, Str8 str, size_t begin);
// Pushes a substring of the given string to the arena, from the index `begin`
// to the index `end` (exclusive).
Str8 PushSubStr8(Arena *arena, Str8 str, size_t begin, size_t end);

#endif  // BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_TESTONLY_OUTPUT_SERVICE_STRING_H_
