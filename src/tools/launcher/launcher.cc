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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "src/tools/launcher/launcher.h"

#include <windows.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace bazel {
namespace launcher {

using std::ifstream;
using std::string;
using std::unordered_map;
using std::vector;
using std::wostringstream;
using std::wstring;
using bazel::tools::cpp::runfiles::Runfiles;

BinaryLauncherBase::BinaryLauncherBase(
    const LaunchDataParser::LaunchInfo& _launch_info,
    const std::wstring& launcher_path, int argc, wchar_t* argv[])
    : launcher_path(launcher_path),
      launch_info(_launch_info),
      workspace_name(GetLaunchInfoByKey(WORKSPACE_NAME)) {
  std::string error;
  runfiles = Runfiles::Create(blaze_util::WstringToCstring(launcher_path), "", &error);
  if (!runfiles) {
    die(L"Failed to find runfiles: %s", error.c_str());
  }
  for (int i = 0; i < argc; i++) {
    commandline_arguments.push_back(argv[i]);
  }
}

wstring BinaryLauncherBase::GetLauncherPath() const { return launcher_path; }

wstring BinaryLauncherBase::GetRunfilesPath() const {
  wstring runfiles_path =
      GetBinaryPathWithExtension(launcher_path) + L".runfiles";
  std::replace(runfiles_path.begin(), runfiles_path.end(), L'/', L'\\');
  return runfiles_path;
}

wstring BinaryLauncherBase::Rlocation(wstring path,
                                      bool has_workspace_name) const {
  // No need to do rlocation if the path is absolute.
  if (blaze_util::IsAbsolute(path)) {
    return path;
  }

  if (path.find(L"../") == 0) {
    // Ignore 'has_workspace_name' when the runfile path is under "../". Such
    // paths already have a workspace name in the next path component. We could
    // append it to this->workspace_name and re-evaluate it, but this is
    // simpler.
    path = path.substr(3);
  } else if (!has_workspace_name) {
    path = this->workspace_name + L"/" + path;
  }

  return blaze_util::CstringToWstring(runfiles->Rlocation(blaze_util::WstringToCstring(path)));
}

wstring BinaryLauncherBase::GetLaunchInfoByKey(const string& key) {
  auto item = launch_info.find(key);
  if (item == launch_info.end()) {
    die(L"Cannot find key \"%hs\" from launch data.\n", key.c_str());
  }
  return item->second;
}

const vector<wstring>& BinaryLauncherBase::GetCommandlineArguments() const {
  return this->commandline_arguments;
}

void BinaryLauncherBase::CreateCommandLine(
    CmdLine* result, const wstring& executable,
    const vector<wstring>& arguments) const {
  wostringstream cmdline;
  cmdline << L'\"' << executable << L'\"';
  for (const auto& s : arguments) {
    cmdline << L' ' << s;
  }

  wstring cmdline_str = cmdline.str();
  if (cmdline_str.size() >= MAX_CMDLINE_LENGTH) {
    die(L"Command line too long: %s", cmdline_str.c_str());
  }

  // Copy command line into a mutable buffer.
  // CreateProcess is allowed to mutate its command line argument.
  wcsncpy(result->cmdline, cmdline_str.c_str(), MAX_CMDLINE_LENGTH - 1);
  result->cmdline[MAX_CMDLINE_LENGTH - 1] = 0;
}

bool BinaryLauncherBase::PrintLauncherCommandLine(
    const wstring& executable, const vector<wstring>& arguments) const {
  bool has_print_cmd_flag = false;
  for (const auto& arg : arguments) {
    has_print_cmd_flag |= (arg == L"--print_launcher_command");
  }
  if (has_print_cmd_flag) {
    wprintf(L"%s\n", executable.c_str());
    for (const auto& arg : arguments) {
      wprintf(L"%s\n", arg.c_str());
    }
  }
  return has_print_cmd_flag;
}

ExitCode BinaryLauncherBase::LaunchProcess(const wstring& executable,
                                           const vector<wstring>& arguments,
                                           bool suppressOutput) const {
  if (PrintLauncherCommandLine(executable, arguments)) {
    return 0;
  }
  for (auto& envVar : runfiles->EnvVars()) {
    SetEnv(blaze_util::CstringToWstring(envVar.first),
           blaze_util::CstringToWstring(envVar.second));
  }
  CmdLine cmdline;
  CreateCommandLine(&cmdline, executable, arguments);
  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFOW startupInfo = {0};
  startupInfo.cb = sizeof(startupInfo);
  BOOL ok = CreateProcessW(
      /* lpApplicationName */ nullptr,
      /* lpCommandLine */ cmdline.cmdline,
      /* lpProcessAttributes */ nullptr,
      /* lpThreadAttributes */ nullptr,
      /* bInheritHandles */ FALSE,
      /* dwCreationFlags */
                              suppressOutput
                              ? CREATE_NO_WINDOW  // no console window => no output
                              : 0,
      /* lpEnvironment */ nullptr,
      /* lpCurrentDirectory */ nullptr,
      /* lpStartupInfo */ &startupInfo,
      /* lpProcessInformation */ &processInfo);
  if (!ok) {
    PrintError(L"Cannot launch process: %s\nReason: %hs", cmdline.cmdline,
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
