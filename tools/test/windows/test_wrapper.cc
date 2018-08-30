// Copyright 2018 The Bazel Authors. All rights reserved.
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

// Test wrapper implementation for Windows.
// Design:
// https://github.com/laszlocsomor/proposals/blob/win-test-runner/designs/2018-07-18-windows-native-test-runner.md

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include <functional>
#include <memory>
#include <string>

#include "src/main/cpp/util/path_platform.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

class Defer {
 public:
  explicit Defer(std::function<void()> f) : f_(f) {}
  ~Defer() { f_(); }

 private:
  std::function<void()> f_;
};

void LogError(const int line, const char* msg) {
  fprintf(stderr, "ERROR(" __FILE__ ":%d) %s\n", line, msg);
}

void LogErrorWithValue(const int line, const char* msg, DWORD error_code) {
  fprintf(stderr, "ERROR(" __FILE__ ":%d) error code: %d (0x%08x): %s\n",
          line, error_code, error_code, msg);
}

bool GetEnv(const char* name, std::string* result) {
  char value[MAX_PATH];
  DWORD size = GetEnvironmentVariableA(name, value, MAX_PATH);
  if (size < MAX_PATH) {
    *result = value;
    return true;
  } else if (size >= MAX_PATH) {
    std::unique_ptr<char[]> value_big(new char[size]);
    GetEnvironmentVariableA(name, value_big.get(), size);
    *result = value_big.get();
    return true;
  } else {
    return false;
  }
}

inline void PrintTestLogStartMarker() {
  // This header marks where --test_output=streamed will start being printed.
  printf("-----------------------------------------------------------------------------\n");
}

inline bool GetWorkspaceName(std::string* result) {
  return GetEnv("TEST_WORKSPACE", result) && !result->empty();
}

inline void StripLeadingDotSlash(std::string* s) {
  if (s->size() >= 2 && (*s)[0] == '.' && (*s)[1] == '/') {
    *s = s->substr(2);
  }
}

bool FindTestBinary(const std::string& argv0, std::string test_path,
                    std::wstring* result) {
  if (!blaze_util::IsAbsolute(test_path)) {
    std::string error;
    std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
        bazel::tools::cpp::runfiles::Runfiles::Create(argv0, &error));
    if (runfiles == nullptr) {
      LogError(__LINE__, "Failed to load runfiles");
      return false;
    }

    std::string workspace;
    if (!GetWorkspaceName(&workspace)) {
      LogError(__LINE__, "Failed to read %TEST_WORKSPACE%");
      return false;
    }

    StripLeadingDotSlash(&test_path);
    test_path = runfiles->Rlocation(workspace + "/" + test_path);
  }

  std::string error;
  if (!blaze_util::AsWindowsPath(test_path, result, &error)) {
    LogError(__LINE__, error.c_str());
    return false;
  }
  return true;
}

bool StartSubprocess(const wchar_t* path, HANDLE* process) {
  // kMaxCmdline value: see lpCommandLine parameter of CreateProcessW.
  static constexpr size_t kMaxCmdline = 32768;

  std::unique_ptr<WCHAR[]> cmdline(new WCHAR[kMaxCmdline]);
  size_t len = wcslen(path);
  wcsncpy(cmdline.get(), path, len + 1);

  PROCESS_INFORMATION processInfo;
  STARTUPINFOW startupInfo = {0};

  if (CreateProcessW(NULL, cmdline.get(), NULL, NULL, FALSE, 0, NULL, NULL,
                     &startupInfo, &processInfo) != 0) {
    CloseHandle(processInfo.hThread);
    *process = processInfo.hProcess;
    return true;
  } else {
    LogErrorWithValue(__LINE__, "CreateProcessW failed", GetLastError());
    return false;
  }
}

int WaitForSubprocess(HANDLE process) {
  DWORD result = WaitForSingleObject(process, INFINITE);
  switch (result) {
    case WAIT_OBJECT_0:
      {
        DWORD exit_code;
        if (!GetExitCodeProcess(process, &exit_code)) {
          LogErrorWithValue(__LINE__, "GetExitCodeProcess failed",
                            GetLastError());
          return 1;
        }
        return exit_code;
      }
    case WAIT_FAILED:
      LogErrorWithValue(__LINE__, "WaitForSingleObject failed", GetLastError());
      return 1;
    default:
      LogErrorWithValue(
          __LINE__, "WaitForSingleObject returned unexpected result", result);
      return 1;
  }
}

}  // namespace


int main(int argc, char** argv) {
  // TODO(laszlocsomor): Implement the functionality described in
  // https://github.com/laszlocsomor/proposals/blob/win-test-runner/designs/2018-07-18-windows-native-test-runner.md

  const char* argv0 = argv[0];
  argc--;
  argv++;
  bool suppress_output = false;
  if (argc > 0 && strcmp(argv[0], "--no_echo") == 0) {
    // Don't print anything to stdout in this special case.
    // Currently needed for persistent test runner.
    suppress_output = true;
    argc--;
    argv++;
  } else {
    std::string test_target;
    if (!GetEnv("TEST_TARGET", &test_target)) {
      LogError(__LINE__, "Failed to read %TEST_TARGET%");
      return 1;
    }
    printf("Executing tests from %s\n", test_target.c_str());
  }

  if (argc < 1) {
    LogError(__LINE__, "Usage: $0 [--no_echo] <test_path> [test_args...]");
    return 1;
  }

  if (!suppress_output) {
    PrintTestLogStartMarker();
  }

  const char* test_path_arg = argv[0];
  std::wstring test_path;
  if (!FindTestBinary(argv0, test_path_arg, &test_path)) {
    return 1;
  }

  HANDLE process;
  if (!StartSubprocess(test_path.c_str(), &process)) {
    return 1;
  }
  Defer close_process([process]() { CloseHandle(process); });

  return WaitForSubprocess(process);
}
