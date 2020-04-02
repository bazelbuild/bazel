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

#include "src/tools/launcher/python_launcher.h"

#include <string>
#include <vector>

#include "src/main/native/windows/process.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::vector;
using std::wstring;

static constexpr const char* PYTHON_BIN_PATH = "python_bin_path";
static constexpr const char* USE_ZIP_FILE = "use_zip_file";

ExitCode PythonBinaryLauncher::Launch() {
  wstring python_binary = this->GetLaunchInfoByKey(PYTHON_BIN_PATH);

  // There are three kinds of values for `python_binary`:
  // 1. An absolute path to a system interpreter. This is the case if
  // `--python_path` is set by the
  //    user, or if a `py_runtime` is used that has `interpreter_path` set.
  // 2. A runfile path to an in-workspace interpreter. This is the case if a
  // `py_runtime` is used that has `interpreter` set.
  // 3. The special constant, "python". This is the default case if neither of
  // the above apply. Rlocation resolves runfiles paths to absolute paths, and
  // if given an absolute path it leaves it alone, so it's suitable for cases 1
  // and 2.
  if (GetBinaryPathWithoutExtension(python_binary) != L"python") {
    // Rlocation returns the original path if python_binary is an absolute path.
    python_binary = this->Rlocation(python_binary, true);
  }

  // If specified python binary path doesn't exist, then fall back to
  // python.exe and hope it's in PATH.
  if (!DoesFilePathExist(python_binary.c_str())) {
    python_binary = L"python.exe";
  }

  vector<wstring> args = this->GetCommandlineArguments();
  wstring use_zip_file = this->GetLaunchInfoByKey(USE_ZIP_FILE);
  wstring python_file;
  // In case the given binary path is a shortened Windows 8dot3 path, we need to
  // convert it back to its long path form before using it to find the python
  // file.
  wstring full_binary_path = GetWindowsLongPath(args[0]);
  if (use_zip_file == L"1") {
    python_file = GetBinaryPathWithoutExtension(full_binary_path) + L".zip";
  } else {
    python_file = GetBinaryPathWithoutExtension(full_binary_path);
  }

  // Replace the first argument with python file path
  args[0] = python_file;

  for (int i = 1; i < args.size(); i++) {
    args[i] = bazel::windows::WindowsEscapeArg(args[i]);
  }

  return this->LaunchProcess(python_binary, args);
}

}  // namespace launcher
}  // namespace bazel
