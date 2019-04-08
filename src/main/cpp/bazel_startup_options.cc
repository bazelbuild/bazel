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
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/workspace_layout.h"

namespace blaze {

BazelStartupOptions::BazelStartupOptions(
    const WorkspaceLayout *workspace_layout)
    : StartupOptions("Bazel", workspace_layout),
      user_bazelrc_(""),
      use_system_rc(true),
      use_workspace_rc(true),
      use_home_rc(true),
      use_master_bazelrc_(true),
      incompatible_windows_style_arg_escaping(true) {
  RegisterNullaryStartupFlag("home_rc");
  RegisterNullaryStartupFlag("incompatible_windows_style_arg_escaping");
  RegisterNullaryStartupFlag("master_bazelrc");
  RegisterNullaryStartupFlag("system_rc");
  RegisterNullaryStartupFlag("workspace_rc");
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
    user_bazelrc_ = *value;
  } else if (GetNullaryOption(arg, "--system_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --system_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_system_rc = true;
    option_sources["system_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--nosystem_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --nosystem_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_system_rc = false;
    option_sources["system_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--workspace_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --workspace_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_workspace_rc = true;
    option_sources["workspace_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--noworkspace_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --noworkspace_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_workspace_rc = false;
    option_sources["workspace_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--home_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --home_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_home_rc = true;
    option_sources["home_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--nohome_rc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --nohome_rc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_home_rc = false;
    option_sources["home_rc"] = rcfile;
  } else if (GetNullaryOption(arg, "--master_bazelrc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --master_bazelrc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_master_bazelrc_ = true;
    option_sources["blazerc"] = rcfile;
  } else if (GetNullaryOption(arg, "--nomaster_bazelrc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --nomaster_bazelrc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    use_master_bazelrc_ = false;
    option_sources["blazerc"] = rcfile;
  } else if (GetNullaryOption(arg,
                              "--incompatible_windows_style_arg_escaping")) {
    incompatible_windows_style_arg_escaping = true;
    option_sources["incompatible_windows_style_arg_escaping"] = rcfile;
  } else if (GetNullaryOption(arg,
                              "--noincompatible_windows_style_arg_escaping")) {
    incompatible_windows_style_arg_escaping = false;
    option_sources["incompatible_windows_style_arg_escaping"] = rcfile;
  } else {
    *is_processed = false;
    return blaze_exit_code::SUCCESS;
  }

  *is_processed = true;
  return blaze_exit_code::SUCCESS;
}

void BazelStartupOptions::MaybeLogStartupOptionWarnings() const {
  if (ignore_all_rc_files) {
    if (!user_bazelrc_.empty()) {
      BAZEL_LOG(WARNING) << "Value of --bazelrc is ignored, since "
                            "--ignore_all_rc_files is on.";
    }
    if ((use_home_rc) &&
        option_sources.find("home_rc") != option_sources.end()) {
      BAZEL_LOG(WARNING) << "Explicit value of --home_rc is "
                            "ignored, since --ignore_all_rc_files is on.";
    }
    if ((use_system_rc) &&
        option_sources.find("system_rc") != option_sources.end()) {
      BAZEL_LOG(WARNING) << "Explicit value of --system_rc is "
                            "ignored, since --ignore_all_rc_files is on.";
    }
    if ((use_workspace_rc) &&
        option_sources.find("workspace_rc") != option_sources.end()) {
      BAZEL_LOG(WARNING) << "Explicit value of --workspace_rc is "
                            "ignored, since --ignore_all_rc_files is on.";
    }
  }
}

void BazelStartupOptions::AddExtraOptions(
    std::vector<std::string> *result) const {
  if (incompatible_windows_style_arg_escaping) {
    result->push_back("--incompatible_windows_style_arg_escaping");
  } else {
    result->push_back("--noincompatible_windows_style_arg_escaping");
  }
}

}  // namespace blaze
