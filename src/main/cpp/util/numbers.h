// Copyright 2014 Google Inc. All rights reserved.
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
#ifndef DEVTOOLS_BLAZE_MAIN_UTIL_NUMBERS_H_
#define DEVTOOLS_BLAZE_MAIN_UTIL_NUMBERS_H_

#include <string>

typedef signed char int8;
typedef int int32;
typedef long long int64;  // NOLINT

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned long long uint64;  // NOLINT

namespace blaze_util {

using std::string;

bool safe_strto32(const string &text, int *value);

int32 strto32(const char *str, char **endptr, int base);

}  // namespace blaze_util

#endif  // DEVTOOLS_BLAZE_MAIN_UTIL_NUMBERS_H_
