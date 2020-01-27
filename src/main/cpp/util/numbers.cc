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
#include "src/main/cpp/util/numbers.h"

#include <errno.h>  // errno, ERANGE
#include <limits.h>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <limits>

#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::string;

static const int32_t kint32min = static_cast<int32_t>(~0x7FFFFFFF);
static const int32_t kint32max = static_cast<int32_t>(0x7FFFFFFF);

// Represents integer values of digits.
// Uses 36 to indicate an invalid character since we support
// bases up to 36.
static const int8_t kAsciiToInt[256] = {
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,  // 16 36s.
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  36, 36, 36, 36, 36, 36, 36,
  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  36, 36, 36, 36, 36, 36,
  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36 };

// Parse the sign.
inline bool safe_parse_sign(const char** rest, /*inout*/
                            bool* negative_ptr  /*output*/) {
  const char* start = *rest;
  const char* end = start + strlen(start);

  // Consume whitespace.
  while (start < end && ascii_isspace(start[0])) {
    ++start;
  }
  while (start < end && ascii_isspace(end[-1])) {
    --end;
  }
  if (start >= end) {
    return false;
  }

  // Consume sign.
  *negative_ptr = (start[0] == '-');
  if (*negative_ptr || start[0] == '+') {
    ++start;
    if (start >= end) {
      return false;
    }
  }

  *rest = start;
  return true;
}

// Consume digits.
//
// The classic loop:
//
//   for each digit
//     value = value * base + digit
//   value *= sign
//
// The classic loop needs overflow checking.  It also fails on the most
// negative integer, -2147483648 in 32-bit two's complement representation.
//
// My improved loop:
//
//  if (!negative)
//    for each digit
//      value = value * base
//      value = value + digit
//  else
//    for each digit
//      value = value * base
//      value = value - digit
//
// Overflow checking becomes simple.

inline bool safe_parse_positive_int(const char *text, int* value_p) {
  int value = 0;
  const int vmax = std::numeric_limits<int>::max();
  static_assert(vmax > 0, "");
  const int vmax_over_base = vmax / 10;
  const char* start = text;
  const char* end = start + strlen(text);
  // loop over digits
  for (; start < end; ++start) {
    unsigned char c = static_cast<unsigned char>(start[0]);
    int digit = kAsciiToInt[c];
    if (digit >= 10) {
      *value_p = value;
      return false;
    }
    if (value > vmax_over_base) {
      *value_p = vmax;
      return false;
    }
    value *= 10;
    if (value > vmax - digit) {
      *value_p = vmax;
      return false;
    }
    value += digit;
  }
  *value_p = value;
  return true;
}

inline bool safe_parse_negative_int(const char *text, int* value_p) {
  int value = 0;
  const int vmin = std::numeric_limits<int>::min();
  static_assert(vmin < 0, "");
  int vmin_over_base = vmin / 10;
  // 2003 c++ standard [expr.mul]
  // "... the sign of the remainder is implementation-defined."
  // Although (vmin/base)*base + vmin%base is always vmin.
  // 2011 c++ standard tightens the spec but we cannot rely on it.
  if (vmin % 10 > 0) {
    vmin_over_base += 1;
  }
  const char* start = text;
  const char* end = start + strlen(text);
  // loop over digits
  for (; start < end; ++start) {
    unsigned char c = static_cast<unsigned char>(start[0]);
    int digit = kAsciiToInt[c];
    if (digit >= 10) {
      *value_p = value;
      return false;
    }
    if (value < vmin_over_base) {
      *value_p = vmin;
      return false;
    }
    value *= 10;
    if (value < vmin + digit) {
      *value_p = vmin;
      return false;
    }
    value -= digit;
  }
  *value_p = value;
  return true;
}

bool safe_strto32(const string &text, int *value_p) {
  *value_p = 0;
  const char* rest = text.c_str();
  bool negative;
  if (!safe_parse_sign(&rest, &negative)) {
    return false;
  }
  if (!negative) {
    return safe_parse_positive_int(rest, value_p);
  } else {
    return safe_parse_negative_int(rest, value_p);
  }
}

int32_t strto32(const char *str, char **endptr, int base) {
  if (sizeof(int32_t) == sizeof(long)) {  // NOLINT
    return static_cast<int32_t>(strtol(str, endptr, base));  // NOLINT
  }
  const int saved_errno = errno;
  errno = 0;
  const long result = strtol(str, endptr, base);  // NOLINT
  if (errno == ERANGE && result == LONG_MIN) {
    return kint32min;
  } else if (errno == ERANGE && result == LONG_MAX) {
    return kint32max;
  } else if (errno == 0 && result < kint32min) {
    errno = ERANGE;
    return kint32min;
  } else if (errno == 0 && result > kint32max) {
    errno = ERANGE;
    return kint32max;
  }
  if (errno == 0)
    errno = saved_errno;
  return static_cast<int32_t>(result);
}

}  // namespace blaze_util
