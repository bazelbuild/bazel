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

using std::vector;
using std::wstring;

static constexpr const char* PYTHON_BIN_PATH = "python_bin_path";
static constexpr const char* USE_ZIP_FILE = "use_zip_file";
static constexpr const char* WINDOWS_STYLE_ESCAPE_JVM_FLAGS = "escape_args";

ExitCode PythonBinaryLauncher::Launch() {
  wstring python_binary = this->GetLaunchInfoByKey(PYTHON_BIN_PATH);
  // If specified python binary path doesn't exist, then fall back to
  // python.exe and hope it's in PATH.
  if (!DoesFilePathExist(python_binary.c_str())) {
    python_binary = L"python.exe";
  }

  vector<wstring> args = this->GetCommandlineArguments();
  wstring use_zip_file = this->GetLaunchInfoByKey(USE_ZIP_FILE);
  wstring python_file;
  if (use_zip_file == L"1") {
    python_file = GetBinaryPathWithoutExtension(args[0]) + L".zip";
  } else {
    python_file = GetBinaryPathWithoutExtension(args[0]);
  }

  // Replace the first argument with python file path
  args[0] = python_file;

  wstring (*const escape_arg_func)(const wstring&) =
      this->GetLaunchInfoByKey(WINDOWS_STYLE_ESCAPE_JVM_FLAGS) == L"1"
          ? WindowsEscapeArg2
          : WindowsEscapeArg;

  for (int i = 1; i < args.size(); i++) {
    args[i] = escape_arg_func(args[i]);
  }

  return this->LaunchProcess(python_binary, args);
}

}  // namespace launcher
}  // namespace bazel
