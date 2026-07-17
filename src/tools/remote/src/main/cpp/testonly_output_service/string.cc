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

#include "src/tools/remote/src/main/cpp/testonly_output_service/string.h"

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "src/tools/remote/src/main/cpp/testonly_output_service/memory.h"

ParsedUInt32 ParseUInt32(Str8 str) {
  ParsedUInt32 result = {};

  result.valid = true;
  for (size_t i = 0; i < str.len; ++i) {
    uint8_t ch = str.ptr[i];
    if (ch < '0' || ch > '9') {
      result.valid = false;
      break;
    }
    result.value = result.value * 10 + ch - '0';
  }

  return result;
}

Str8 PushStr8(Arena *arena, Str8 str) {
  uint8_t *ptr = PushArray(arena, uint8_t, str.len + 1);
  memcpy(ptr, str.ptr, str.len + 1);
  Str8 result = {ptr, str.len};
  return result;
}

Str8 PushStr8F(Arena *arena, const char *format, ...) {
  constexpr size_t kInitBufferSize = 256;
  size_t buf_len = kInitBufferSize;
  char *buf_ptr = PushArray(arena, char, buf_len);

  va_list args;
  va_start(args, format);
  size_t str_len = vsnprintf(buf_ptr, buf_len, format, args);
  va_end(args);

  if (str_len + 1 <= buf_len) {
    // Free the unused part of the buffer.
    PopArena(arena, buf_len - str_len - 1);
  } else {
    // The buffer was too small. We need to resize it and try again.
    PopArena(arena, buf_len);
    buf_len = str_len + 1;
    buf_ptr = PushArray(arena, char, buf_len);
    va_start(args, format);
    vsnprintf(buf_ptr, buf_len, format, args);
    va_end(args);
  }

  Str8 result = {(uint8_t *)buf_ptr, str_len};
  return result;
}

Str8 PushSubStr8(Arena *arena, Str8 str, size_t begin) {
  Str8 result = PushSubStr8(arena, str, begin, str.len);
  return result;
}

Str8 PushSubStr8(Arena *arena, Str8 str, size_t begin, size_t end) {
  assert(begin <= end && end <= str.len);
  size_t len = end - begin;
  uint8_t *ptr = PushArray(arena, uint8_t, len + 1);
  memcpy(ptr, str.ptr + begin, len);
  Str8 result = {ptr, len};
  return result;
}
