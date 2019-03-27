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

#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/launcher.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::ifstream;
using std::string;
using std::unordered_map;
using std::vector;
using std::wostringstream;
using std::wstring;

static wstring GetRunfilesDir(const wchar_t* argv0) {
  wstring runfiles_dir;
  // If RUNFILES_DIR is already set (probably we are either in a test or in a
  // data dependency) then use it.
  if (GetEnv(L"RUNFILES_DIR", &runfiles_dir)) {
    return runfiles_dir;
  }
  // Otherwise this is probably a top-level non-test binary (e.g. a genrule
  // tool) and should look for its runfiles beside the executable.
  return GetBinaryPathWithExtension(argv0) + L".runfiles";
}

BinaryLauncherBase::BinaryLauncherBase(
    const LaunchDataParser::LaunchInfo& _launch_info, int argc, wchar_t* argv[])
    : launch_info(_launch_info),
      runfiles_dir(GetRunfilesDir(argv[0])),
      workspace_name(GetLaunchInfoByKey(WORKSPACE_NAME)),
      symlink_runfiles_enabled(GetLaunchInfoByKey(SYMLINK_RUNFILES_ENABLED) ==
                               L"1") {
  for (int i = 0; i < argc; i++) {
    commandline_arguments.push_back(argv[i]);
  }

  std::string cargv0(blaze_util::WstringToCstring(argv[0]).get());
  runfiles_.reset(bazel::tools::cpp::runfiles::Runfiles::Create(cargv0));
  if (runfiles_ == nullptr) {
    die(L"Couldn't init runfiles");
  }
}

static bool FindManifestFileImpl(const wchar_t* argv0, wstring* result) {
  // If this binary X runs as the data-dependency of some other binary Y, then
  // X has no runfiles manifest/directory and should use Y's.
  if (GetEnv(L"RUNFILES_MANIFEST_FILE", result) &&
      DoesFilePathExist(result->c_str())) {
    return true;
  }

  wstring directory;
  if (GetEnv(L"RUNFILES_DIR", &directory)) {
    *result = directory + L"/MANIFEST";
    if (DoesFilePathExist(result->c_str())) {
      return true;
    }
  }

  // If this binary X runs by itself (not as a data-dependency of another
  // binary), then look for the manifest in a runfiles directory next to the
  // main binary, then look for it (the manifest) next to the main binary.
  directory = GetBinaryPathWithExtension(argv0) + L".runfiles";
  *result = directory + L"/MANIFEST";
  if (DoesFilePathExist(result->c_str())) {
    return true;
  }

  *result = directory + L"_manifest";
  if (DoesFilePathExist(result->c_str())) {
    return true;
  }

  return false;
}

wstring BinaryLauncherBase::GetRunfilesPath() const {
  wstring runfiles_path =
      GetBinaryPathWithExtension(this->commandline_arguments[0]) + L".runfiles";
  std::replace(runfiles_path.begin(), runfiles_path.end(), L'/', L'\\');
  return runfiles_path;
}

wstring BinaryLauncherBase::Rlocation(wstring path,
                                      bool has_workspace_name) const {
  // No need to do rlocation if the path is absolute.
  if (blaze_util::IsAbsolute(path)) {
    return path;
  }

  if (path.find(L"external/") == 0) {
    // Ignore 'has_workspace_name' when the path is under "external/". Such
    // paths already have a workspace name in the next path component.
    path = path.substr(9);
  } else if (!has_workspace_name) {
    path = this->workspace_name + L"/" + path;
  }

  string cpath = blaze_util::WstringToCstring(path.c_str()).get();
  string crloc = runfiles_->Rlocation(cpath);
  return blaze_util::CstringToWstring(crloc.c_str()).get();
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

  bool manifest_exists = false;
  for (auto i : runfiles_->EnvVars()) {
    // As of 2019-03-27, the envvars list contains "RUNFILES_MANIFEST_FILE" only
    // if its value points to an existing manifest.
    if (!manifest_exists && i.first == "RUNFILES_MANIFEST_FILE") {
      manifest_exists = true;
    }
    SetEnv(
        blaze_util::CstringToWstring(i.first.c_str()).get(),
        blaze_util::CstringToWstring(i.second.c_str()).get());
  }
  // As of 2019-03-27 the list doesn't contain RUNFILES_MANIFEST_ONLY, so we set
  // it manually. 
  SetEnv(L"RUNFILES_MANIFEST_ONLY",
         manifest_exists && !symlink_runfiles_enabled ? L"1" : L"");

  CmdLine cmdline;
  CreateCommandLine(&cmdline, executable, arguments);
  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFOW startupInfo = {0};
  startupInfo.cb = sizeof(startupInfo);
  BOOL ok = CreateProcessW(
      /* lpApplicationName */ NULL,
      /* lpCommandLine */ cmdline.cmdline,
      /* lpProcessAttributes */ NULL,
      /* lpThreadAttributes */ NULL,
      /* bInheritHandles */ FALSE,
      /* dwCreationFlags */
      suppressOutput ? CREATE_NO_WINDOW  // no console window => no output
                     : 0,
      /* lpEnvironment */ NULL,
      /* lpCurrentDirectory */ NULL,
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
