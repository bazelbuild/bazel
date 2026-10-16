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

#include <string>
#include <vector>

#ifdef __APPLE__
#include <sys/syslimits.h>
#include <unistd.h>
#endif

#include "src/main/cpp/option_processor-internal.h"

// On OSX, there apparently is no header that defines this.
#ifndef environ
extern char** environ;
#endif

namespace blaze {

std::vector<std::string> GetProcessedEnv() {
  std::vector<std::string> processed_env;
  for (char** env = environ; *env != nullptr; env++) {
    processed_env.emplace_back(*env);
  }

#ifdef __APPLE__
  for (int key : {_CS_DARWIN_USER_TEMP_DIR, _CS_DARWIN_USER_CACHE_DIR}) {
    char buf[PATH_MAX];
    if (confstr(key, buf, sizeof(buf)) > 0) {
      const char* name = (key == _CS_DARWIN_USER_TEMP_DIR)
                             ? "DARWIN_USER_TEMP_DIR"
                             : "DARWIN_USER_CACHE_DIR";
      processed_env.push_back(std::string(name) + "=" + buf);
    }
  }
#endif

  return processed_env;
}

}  // namespace blaze
