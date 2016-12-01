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

#include "third_party/ijar/platform_utils.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>

#ifdef COMPILER_MSVC
#else  // not COMPILER_MSVC
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // COMPILER_MSVC

#include <string>

namespace devtools_ijar {

bool stat_file(const char* path, Stat* result) {
#ifdef COMPILER_MSVC
  // TODO(laszlocsomor) 2016-12-01: implement this and other methods, in order
  // to close https://github.com/bazelbuild/bazel/issues/2157.
  fprintf(stderr, "Not yet implemented on Windows\n");
  return false;
#else   // not COMPILER_MSVC
  struct stat statst;
  if (stat(path, &statst) < 0) {
    return false;
  }
  result->total_size = statst.st_size;
  result->file_mode = statst.st_mode;
  result->is_directory = (statst.st_mode & S_IFDIR) != 0;
  return true;
#endif  // COMPILER_MSVC
}

}  // namespace devtools_ijar
