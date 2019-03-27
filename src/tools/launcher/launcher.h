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

#ifndef BAZEL_SRC_TOOLS_LAUNCHER_LAUNCHER_H_
#define BAZEL_SRC_TOOLS_LAUNCHER_LAUNCHER_H_

#include <memory>
#include <string>
#include <vector>

#include "src/tools/launcher/util/data_parser.h"

// This project depends on the runfiles library from source, not from
// @bazel_tools, which explains why include path is not ".../runfiles.h"
#include "tools/cpp/runfiles/runfiles_src.h"

namespace bazel {
namespace launcher {

typedef int32_t ExitCode;
static constexpr const char* WORKSPACE_NAME = "workspace_name";
static constexpr const char* SYMLINK_RUNFILES_ENABLED =
    "symlink_runfiles_enabled";

// The maximum length of lpCommandLine is 32768 characters.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms682425(v=vs.85).aspx
static const int MAX_CMDLINE_LENGTH = 32768;

struct CmdLine {
  wchar_t cmdline[MAX_CMDLINE_LENGTH];
};

class BinaryLauncherBase {
 public:
  BinaryLauncherBase(const LaunchDataParser::LaunchInfo& launch_info, int argc,
                     wchar_t* argv[]);

  virtual ~BinaryLauncherBase() = default;

  // Get launch information based on a launch info key.
  std::wstring GetLaunchInfoByKey(const std::string& key);

  // Get the original command line arguments passed to this binary.
  const std::vector<std::wstring>& GetCommandlineArguments() const;

  // Map a runfile path to its absolute path.
  //
  // 'has_workspace_name' indicates whether 'path' already starts with the
  // runfile's workspace name. (This is implicitly true when 'path' is under
  // "external/".) If the path does not have a workspace name (and does not
  // start with "external/"), this method prepends the main repository's name to
  // it before looking up the runfile.
  std::wstring Rlocation(std::wstring path,
                         bool has_workspace_name = false) const;

  // Lauch a process with given executable and command line arguments.
  // If --print_launcher_command exists in arguments, then we print the full
  // command line instead of launching the real process.
  //
  // executable: the binary to be executed.
  // arguments:  the command line arguments to be passed to the executable,
  //             it doesn't include the executable itself.
  //             The arguments are expected to be quoted if having spaces.
  ExitCode LaunchProcess(const std::wstring& executable,
                         const std::vector<std::wstring>& arguments,
                         bool suppressOutput = false) const;

  // A launch function to be implemented for a specific language.
  virtual ExitCode Launch() = 0;

  // Return the runfiles directory of this binary.
  //
  // The method appends ".exe.runfiles" to the first command line argument,
  // converts forward slashes to back slashes, then returns that.
  std::wstring GetRunfilesPath() const;

 private:
  // A map to store all the launch information.
  const LaunchDataParser::LaunchInfo& launch_info;

  // Path to the runfiles directory, if one exists.
  const std::wstring runfiles_dir;

  // The commandline arguments received.
  // The first argument is the path of this launcher itself.
  std::vector<std::wstring> commandline_arguments;

  // The workspace name of the repository this target belongs to.
  const std::wstring workspace_name;

  // If symlink runfiles tree is enabled, this value is true.
  const bool symlink_runfiles_enabled;

  // If --print_launcher_command is presented in arguments,
  // then print the command line.
  //
  // Return true if command line is printed.
  bool PrintLauncherCommandLine(
      const std::wstring& executable,
      const std::vector<std::wstring>& arguments) const;

  // Create a command line to be passed to Windows CreateProcessA API.
  //
  // executable: the binary to be executed.
  // arguments:  the command line arguments to be passed to the executable,
  //             it doesn't include the executable itself.
  void CreateCommandLine(CmdLine* result, const std::wstring& executable,
                         const std::vector<std::wstring>& arguments) const;

  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles_;
};

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_LAUNCHER_H_
