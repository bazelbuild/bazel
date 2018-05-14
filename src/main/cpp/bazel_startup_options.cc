// Copyright 2018 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/bazel_startup_options.h"

#include <cassert>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/workspace_layout.h"

namespace blaze {

BazelStartupOptions::BazelStartupOptions(
    const WorkspaceLayout *workspace_layout)
    : StartupOptions("Bazel", workspace_layout) {
  RegisterNullaryStartupFlag("master_bazelrc");
  RegisterUnaryStartupFlag("bazelrc");
}

blaze_exit_code::ExitCode BazelStartupOptions::ProcessArgExtra(
    const char *arg, const char *next_arg, const std::string &rcfile,
    const char **value, bool *is_processed, std::string *error) {
  assert(value);
  assert(is_processed);

  if ((*value = GetUnaryOption(arg, next_arg, "--bazelrc")) != nullptr) {
    if (!rcfile.empty()) {
      *error = "Can't specify --bazelrc in the .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
  } else if (GetNullaryOption(arg, "--nomaster_bazelrc") ||
             GetNullaryOption(arg, "--master_bazelrc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --[no]master_bazelrc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["blazerc"] = rcfile;
  } else {
    *is_processed = false;
    return blaze_exit_code::SUCCESS;
  }

  *is_processed = true;
  return blaze_exit_code::SUCCESS;
}

}  // namespace blaze
