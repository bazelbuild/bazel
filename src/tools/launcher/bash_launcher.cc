// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include <sstream>
#include <string>
#include <vector>

#include "src/tools/launcher/bash_launcher.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::vector;
using std::wostringstream;
using std::wstring;

static constexpr const char* BASH_BIN_PATH = "bash_bin_path";

ExitCode BashBinaryLauncher::Launch() {
  wstring bash_binary = this->GetLaunchInfoByKey(BASH_BIN_PATH);

  // If bash_binary is already "bash" or "bash.exe", that means we want to
  // rely on the shell binary in PATH, no need to do Rlocation.
  if (GetBinaryPathWithoutExtension(bash_binary) != L"bash") {
    // Rlocation returns the original path if bash_binary is an absolute path.
    bash_binary = this->Rlocation(bash_binary, true);
  }

  if (DoesFilePathExist(bash_binary.c_str())) {
    wstring bash_bin_dir = GetParentDirFromPath(bash_binary);
    wstring path_env;
    GetEnv(L"PATH", &path_env);
    path_env = bash_bin_dir + L";" + path_env;
    SetEnv(L"PATH", path_env);
  } else {
    // If specified bash binary path doesn't exist, then fall back to
    // bash.exe and hope it's in PATH.
    bash_binary = L"bash.exe";
  }

  vector<wstring> origin_args = this->GetCommandlineArguments();
  wostringstream bash_command;
  // In case the given binary path is a shortened Windows 8dot3 path, we need to
  // convert it back to its long path form before using it to find the bash main
  // file.
  wstring full_binary_path = GetWindowsLongPath(origin_args[0]);
  wstring bash_main_file = GetBinaryPathWithoutExtension(full_binary_path);
  bash_command << BashEscapeArg(bash_main_file);
  for (int i = 1; i < origin_args.size(); i++) {
    bash_command << L' ';
    bash_command << BashEscapeArg(origin_args[i]);
  }

  vector<wstring> args;
  args.push_back(L"-c");
  args.push_back(BashEscapeArg(bash_command.str()));
  return this->LaunchProcess(bash_binary, args);
}

}  // namespace launcher
}  // namespace bazel
