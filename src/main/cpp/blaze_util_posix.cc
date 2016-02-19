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

#include <limits.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;

using std::string;
using std::vector;

void ExecuteProgram(const string &exe, const vector<string> &args_vector) {
  if (VerboseLogging()) {
    string dbg;
    for (const auto &s : args_vector) {
      dbg.append(s);
      dbg.append(" ");
    }

    char cwd[PATH_MAX] = {};
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "getcwd() failed");
    }

    fprintf(stderr, "Invoking binary %s in %s:\n  %s\n", exe.c_str(), cwd,
            dbg.c_str());
  }

  // Copy to a char* array for execv:
  int n = args_vector.size();
  const char **argv = new const char *[n + 1];
  for (int i = 0; i < n; ++i) {
    argv[i] = args_vector[i].c_str();
  }
  argv[n] = NULL;

  execv(exe.c_str(), const_cast<char **>(argv));
}

std::string ConvertPath(const std::string &path) { return path; }

std::string ListSeparator() { return ":"; }

bool SymlinkDirectories(const string &target, const string &link) {
  return symlink(target.c_str(), link.c_str()) == 0;
}
}   // namespace blaze.
