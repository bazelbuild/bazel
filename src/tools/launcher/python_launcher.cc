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

#include <string>
#include <vector>

#include "src/tools/launcher/python_launcher.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::string;
using std::vector;

static constexpr const char* PYTHON_BIN_PATH = "python_bin_path";

ExitCode PythonBinaryLauncher::Launch() {
  string python_binary = this->GetLaunchInfoByKey(PYTHON_BIN_PATH);
  // If specified python binary path doesn't exist, then fall back to
  // python.exe and hope it's in PATH.
  if (!DoesFilePathExist(python_binary.c_str())) {
    python_binary = "python.exe";
  }

  vector<string> args = this->GetCommandlineArguments();
  string python_zip_file = GetBinaryPathWithoutExtension(args[0]) + ".zip";

  // Replace the first argument with python zip file path
  args[0] = python_zip_file;

  // Escape arguments that has spaces
  for (int i = 1; i < args.size(); i++) {
    args[i] = GetEscapedArgument(args[i], /*escape_backslash = */ false);
  }

  return this->LaunchProcess(python_binary, args);
}

}  // namespace launcher
}  // namespace bazel
