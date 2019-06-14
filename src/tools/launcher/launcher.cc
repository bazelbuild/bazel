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

#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
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
      manifest_file(FindManifestFile(argv[0])),
      runfiles_dir(GetRunfilesDir(argv[0])),
      workspace_name(GetLaunchInfoByKey(WORKSPACE_NAME)),
      symlink_runfiles_enabled(GetLaunchInfoByKey(SYMLINK_RUNFILES_ENABLED) ==
                               L"1") {
  for (int i = 0; i < argc; i++) {
    commandline_arguments.push_back(argv[i]);
  }
  // Prefer to use the runfiles manifest, if it exists, but otherwise the
  // runfiles directory will be used by default. On Windows, the manifest is
  // used locally, and the runfiles directory is used remotely.
  if (!manifest_file.empty()) {
    ParseManifestFile(&manifest_file_map, manifest_file);
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

wstring BinaryLauncherBase::FindManifestFile(const wchar_t* argv0) {
  wstring manifest_file;
  if (!FindManifestFileImpl(argv0, &manifest_file)) {
    return L"";
  }
  // The path will be set as the RUNFILES_MANIFEST_FILE envvar and used by the
  // shell script, so let's convert backslashes to forward slashes.
  std::replace(manifest_file.begin(), manifest_file.end(), '\\', '/');
  return manifest_file;
}

wstring BinaryLauncherBase::GetRunfilesPath() const {
  wstring runfiles_path =
      GetBinaryPathWithExtension(this->commandline_arguments[0]) + L".runfiles";
  std::replace(runfiles_path.begin(), runfiles_path.end(), L'/', L'\\');
  return runfiles_path;
}

void BinaryLauncherBase::ParseManifestFile(ManifestFileMap* manifest_file_map,
                                           const wstring& manifest_path) {
  ifstream manifest_file(AsAbsoluteWindowsPath(manifest_path.c_str()).c_str());

  if (!manifest_file) {
    die(L"Couldn't open MANIFEST file: %s", manifest_path.c_str());
  }

  string line;
  while (getline(manifest_file, line)) {
    size_t space_pos = line.find_first_of(' ');
    if (space_pos == string::npos) {
      die(L"Wrong MANIFEST format at line: %s", line.c_str());
    }
    wstring wline = blaze_util::CstringToWstring(line);
    wstring key = wline.substr(0, space_pos);
    wstring value = wline.substr(space_pos + 1);
    manifest_file_map->insert(make_pair(key, value));
  }
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

  // If the manifest file map is empty, then we're using the runfiles directory
  // instead.
  if (manifest_file_map.empty()) {
    return runfiles_dir + L"/" + path;
  }

  auto entry = manifest_file_map.find(path);
  if (entry == manifest_file_map.end()) {
    die(L"Rlocation failed on %s, path doesn't exist in MANIFEST file",
        path.c_str());
  }
  return entry->second;
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
  // Set RUNFILES_DIR if:
  //   1. Symlink runfiles tree is enabled, or
  //   2. We couldn't find manifest file (which probably means we are running
  //   remotely).
  // Otherwise, set RUNFILES_MANIFEST_ONLY and RUNFILES_MANIFEST_FILE
  if (symlink_runfiles_enabled || manifest_file.empty()) {
    SetEnv(L"RUNFILES_DIR", runfiles_dir);
  } else {
    SetEnv(L"RUNFILES_MANIFEST_ONLY", L"1");
    SetEnv(L"RUNFILES_MANIFEST_FILE", manifest_file);
  }
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
