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

#ifndef THIRD_PARTY_IJAR_PLATFORM_UTILS_H_
#define THIRD_PARTY_IJAR_PLATFORM_UTILS_H_

#include <stdlib.h>

#include <string>

#include "third_party/ijar/common.h"

namespace devtools_ijar {

// Platform-independent stat data.
struct Stat {
  // Total size of the file in bytes.
  int total_size;
  // The Unix file mode from the stat.st_mode field.
  mode_t file_mode;
  // True if this is a directory.
  bool is_directory;
};

// Writes stat data into `result` about the file under `path`.
// Returns true upon success: file is found and can be stat'ed.
bool stat_file(const char* path, Stat* result);

}  // namespace devtools_ijar

#endif  // THIRD_PARTY_IJAR_PLATFORM_UTILS_H_
