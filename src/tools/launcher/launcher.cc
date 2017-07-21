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
    : launch_info(_launch_info) {
  this->workspace_name = GetLaunchInfoByKey(WORKSPACE_NAME);
  for (int i = 0; i < argc; i++) {
    this->commandline_arguments.push_back(argv[i]);
  }
  ParseManifestFile(&this->manifest_file_map, FindManifestFile());
}

string BinaryLauncherBase::FindManifestFile() const {
  // Get the name of the binary
  string binary = GetBinaryPathWithoutExtension(this->commandline_arguments[0]);

  // Try to find <path to binary>.runfiles/MANIFEST
  string manifest_file = binary + ".runfiles\\MANIFEST";
  if (DoesFilePathExist(manifest_file)) {
    return manifest_file;
  }

  // Also try to check if <path to binary>.runfiles_manifest exists
  manifest_file = binary + ".runfiles_manifest";
  if (DoesFilePathExist(manifest_file)) {
    return manifest_file;
  }

  die("Couldn't find MANIFEST file %s.runfiles\\", binary.c_str());
}

void BinaryLauncherBase::ParseManifestFile(ManifestFileMap* manifest_file_map,
                                           const string& manifest_path) {
  ifstream manifest_file(manifest_path.c_str());

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

string BinaryLauncherBase::Rlocation(const string& path) const {
  auto entry = manifest_file_map.find(this->workspace_name + "/" + path);
  if (entry == manifest_file_map.end()) {
    die("Rlocation failed on %s, path doesn't exist in MANIFEST file",
        path.c_str());
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
  bool first = true;
  for (const auto& s : arguments) {
    cmdline << ' ' << GetEscapedArgument(s);
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

ExitCode BinaryLauncherBase::LaunchProcess(
    const string& executable, const vector<string>& arguments) const {
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
    PrintError("Cannot launch process:\n%s", GetLastErrorString().c_str());
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
