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
// TODO(b/32967056) die() and pdie() are really error statements with an exit;
//    these can be removed once logging is on by default.

#ifndef BAZEL_SRC_MAIN_CPP_UTIL_ERRORS_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_ERRORS_H_

#include <string>
#include "src/main/cpp/util/port.h"

namespace blaze_util {

// Returns the last error as a platform-specific error message.
// The string will also contain the platform-specific error code itself
// (which is `errno` on Linux/Darwin, and `GetLastError()` on Windows).
std::string GetLastErrorString();

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_ERRORS_H_
