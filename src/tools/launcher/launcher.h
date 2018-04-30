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

#include <string>
#include <unordered_map>
#include <vector>

#include "src/tools/launcher/util/data_parser.h"

namespace bazel {
namespace launcher {

typedef int32_t ExitCode;
static constexpr const char* WORKSPACE_NAME = "workspace_name";

// The maximum length of lpCommandLine is 32768 characters.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms682425(v=vs.85).aspx
static const int MAX_CMDLINE_LENGTH = 32768;

struct CmdLine {
  char cmdline[MAX_CMDLINE_LENGTH];
};

class BinaryLauncherBase {
  typedef std::unordered_map<std::string, std::string> ManifestFileMap;

 public:
  BinaryLauncherBase(const LaunchDataParser::LaunchInfo& launch_info, int argc,
                     char* argv[]);

  // Get launch information based on a launch info key.
  std::string GetLaunchInfoByKey(const std::string& key);

  // Get the original command line arguments passed to this binary.
  const std::vector<std::string>& GetCommandlineArguments() const;

  // Map a runfile path to its absolute path.
  //
  // If need_workspace_name is true, then this method prepend workspace name to
  // path before doing rlocation.
  // If need_workspace_name is false, then this method uses path directly.
  // The default value of need_workspace_name is true.
  std::string Rlocation(const std::string& path,
                        bool need_workspace_name = true) const;

  // Lauch a process with given executable and command line arguments.
  // If --print_launcher_command exists in arguments, then we print the full
  // command line instead of launching the real process.
  //
  // exectuable: the binary to be executed.
  // arguments:  the command line arguments to be passed to the exectuable,
  //             it doesn't include the exectuable itself.
  //             The arguments are expected to be quoted if having spaces.
  ExitCode LaunchProcess(const std::string& executable,
                         const std::vector<std::string>& arguments,
                         bool suppressOutput = false) const;

  // A launch function to be implemented for a specific language.
  virtual ExitCode Launch() = 0;

  // Return the runfiles directory of this binary.
  //
  // The method appends ".exe.runfiles" to the first command line argument,
  // converts forward slashes to back slashes, then returns that.
  std::string GetRunfilesPath() const;

 private:
  // A map to store all the launch information.
  const LaunchDataParser::LaunchInfo& launch_info;

  // Absolute path to the runfiles manifest file, if one exists.
  const std::string manifest_file;

  // Path to the runfiles directory, if one exists.
  const std::string runfiles_dir;

  // The commandline arguments recieved.
  // The first argument is the path of this launcher itself.
  std::vector<std::string> commandline_arguments;

  // The workspace name of the repository this target belongs to.
  const std::string workspace_name;

  // A map to store all entries of the manifest file.
  std::unordered_map<std::string, std::string> manifest_file_map;

  // If --print_launcher_command is presented in arguments,
  // then print the command line.
  //
  // Return true if command line is printed.
  bool PrintLauncherCommandLine(
      const std::string& executable,
      const std::vector<std::string>& arguments) const;

  // Create a command line to be passed to Windows CreateProcessA API.
  //
  // exectuable: the binary to be executed.
  // arguments:  the command line arguments to be passed to the exectuable,
  //             it doesn't include the exectuable itself.
  void CreateCommandLine(CmdLine* result, const std::string& executable,
                         const std::vector<std::string>& arguments) const;

  // Find manifest file of the binary.
  //
  // Expect the manifest file to be at
  //    1. <path>/<to>/<binary>/<target_name>.runfiles/MANIFEST
  // or 2. <path>/<to>/<binary>/<target_name>.runfiles_manifest
  static std::string FindManifestFile(const char* argv0);

  // Parse manifest file into a map
  static void ParseManifestFile(ManifestFileMap* manifest_file_map,
                                const std::string& manifest_path);
};

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_LAUNCHER_H_
