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

#ifndef BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_

#include <string>

namespace blaze_util {

// Checks each element of the PATH variable for executable. If none is found, ""
// is returned.  Otherwise, the full path to executable is returned. Can die if
// looking up PATH fails.
std::string Which(const std::string &executable);

// Returns true if this path exists.
bool PathExists(const std::string& path);

// Returns true if the path exists and can be accessed to read/write as desired.
//
// If `exec` is true and the path refers to a file, it means the file must be
// executable; if the path is a directory, it means the directory must be
// openable.
bool CanAccess(const std::string& path, bool read, bool write, bool exec);

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_
