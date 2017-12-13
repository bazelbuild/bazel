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

#include <windows.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/tools/launcher/launcher.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::ifstream;
using std::ostringstream;
using std::string;
using std::unordered_map;
using std::vector;

BinaryLauncherBase::BinaryLauncherBase(
    const LaunchDataParser::LaunchInfo& _launch_info, int argc, char* argv[])
    : launch_info(_launch_info),
      manifest_file(FindManifestFile(argv[0])),
      workspace_name(GetLaunchInfoByKey(WORKSPACE_NAME)) {
  for (int i = 0; i < argc; i++) {
    this->commandline_arguments.push_back(argv[i]);
  }
  ParseManifestFile(&this->manifest_file_map, this->manifest_file);
}

string BinaryLauncherBase::FindManifestFile(const char* argv0) {
  // Get the name of the binary
  string binary = GetBinaryPathWithExtension(argv0);

  // The path will be set as the RUNFILES_MANIFEST_FILE envvar and used by the
  // shell script, so let's convert backslashes to forward slashes.
  std::replace(binary.begin(), binary.end(), '\\', '/');

  // Try to find <path to binary>.runfiles/MANIFEST
  string manifest_file = binary + ".runfiles/MANIFEST";
  if (DoesFilePathExist(manifest_file.c_str())) {
    return manifest_file;
  }

  // Also try to check if <path to binary>.runfiles_manifest exists
  manifest_file = binary + ".runfiles_manifest";
  if (DoesFilePathExist(manifest_file.c_str())) {
    return manifest_file;
  }

  die("Couldn't find MANIFEST file under %s.runfiles\\", binary.c_str());
}

string BinaryLauncherBase::GetRunfilesPath() const {
  string runfiles_path =
      GetBinaryPathWithExtension(this->commandline_arguments[0]) + ".runfiles";
  std::replace(runfiles_path.begin(), runfiles_path.end(), '/', '\\');
  return runfiles_path;
}

void BinaryLauncherBase::ParseManifestFile(ManifestFileMap* manifest_file_map,
                                           const string& manifest_path) {
  ifstream manifest_file(AsAbsoluteWindowsPath(manifest_path.c_str()).c_str());

  if (!manifest_file) {
    die("Couldn't open MANIFEST file: %s", manifest_path.c_str());
  }

  string line;
  while (getline(manifest_file, line)) {
    size_t space_pos = line.find_first_of(' ');
    if (space_pos == string::npos) {
      die("Wrong MANIFEST format at line: %s", line.c_str());
    }
    string key = line.substr(0, space_pos);
    string value = line.substr(space_pos + 1);
    manifest_file_map->insert(make_pair(key, value));
  }
}

string BinaryLauncherBase::Rlocation(const string& path,
                                     bool need_workspace_name) const {
  string query_path = path;
  if (need_workspace_name) {
    query_path = this->workspace_name + "/" + path;
  }
  auto entry = manifest_file_map.find(query_path);
  if (entry == manifest_file_map.end()) {
    die("Rlocation failed on %s, path doesn't exist in MANIFEST file",
        query_path.c_str());
  }
  return entry->second;
}

string BinaryLauncherBase::GetLaunchInfoByKey(const string& key) {
  auto item = launch_info.find(key);
  if (item == launch_info.end()) {
    die("Cannot find key \"%s\" from launch data.\n", key.c_str());
  }
  return item->second;
}

const vector<string>& BinaryLauncherBase::GetCommandlineArguments() const {
  return this->commandline_arguments;
}

void BinaryLauncherBase::CreateCommandLine(
    CmdLine* result, const string& executable,
    const vector<string>& arguments) const {
  ostringstream cmdline;
  cmdline << '\"' << executable << '\"';
  for (const auto& s : arguments) {
    cmdline << ' ' << s;
  }

  string cmdline_str = cmdline.str();
  if (cmdline_str.size() >= MAX_CMDLINE_LENGTH) {
    die("Command line too long: %s", cmdline_str.c_str());
  }

  // Copy command line into a mutable buffer.
  // CreateProcess is allowed to mutate its command line argument.
  strncpy(result->cmdline, cmdline_str.c_str(), MAX_CMDLINE_LENGTH - 1);
  result->cmdline[MAX_CMDLINE_LENGTH - 1] = 0;
}

bool BinaryLauncherBase::PrintLauncherCommandLine(
    const string& executable, const vector<string>& arguments) const {
  bool has_print_cmd_flag = false;
  for (const auto& arg : arguments) {
    has_print_cmd_flag |= (arg == "--print_launcher_command");
  }
  if (has_print_cmd_flag) {
    printf("%s\n", executable.c_str());
    for (const auto& arg : arguments) {
      printf("%s\n", arg.c_str());
    }
  }
  return has_print_cmd_flag;
}

ExitCode BinaryLauncherBase::LaunchProcess(
    const string& executable, const vector<string>& arguments) const {
  if (PrintLauncherCommandLine(executable, arguments)) {
    return 0;
  }
  SetEnv("RUNFILES_MANIFEST_ONLY", "1");
  SetEnv("RUNFILES_MANIFEST_FILE", manifest_file);
  CmdLine cmdline;
  CreateCommandLine(&cmdline, executable, arguments);
  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFOA startupInfo = {0};
  startupInfo.cb = sizeof(startupInfo);
  BOOL ok = CreateProcessA(
      /* lpApplicationName */ NULL,
      /* lpCommandLine */ cmdline.cmdline,
      /* lpProcessAttributes */ NULL,
      /* lpThreadAttributes */ NULL,
      /* bInheritHandles */ FALSE,
      /* dwCreationFlags */ 0,
      /* lpEnvironment */ NULL,
      /* lpCurrentDirectory */ NULL,
      /* lpStartupInfo */ &startupInfo,
      /* lpProcessInformation */ &processInfo);
  if (!ok) {
    PrintError("Cannot launch process: %s\nReason: %s",
               cmdline.cmdline,
               GetLastErrorString().c_str());
    return GetLastError();
  }
  WaitForSingleObject(processInfo.hProcess, INFINITE);
  ExitCode exit_code;
  GetExitCodeProcess(processInfo.hProcess,
                     reinterpret_cast<LPDWORD>(&exit_code));
  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);
  return exit_code;
}

}  // namespace launcher
}  // namespace bazel
