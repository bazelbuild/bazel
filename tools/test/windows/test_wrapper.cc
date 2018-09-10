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

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
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
  fprintf(stderr, "ERROR(" __FILE__ ":%d) error code: %d (0x%08x): %s\n", line,
          error_code, error_code, msg);
}

void LogErrorWithArgAndValue(const int line, const char* msg, const char* arg,
                             DWORD error_code) {
  fprintf(stderr,
          "ERROR(" __FILE__ ":%d) error code: %d (0x%08x), argument: %s: %s\n",
          line, error_code, error_code, arg, msg);
}

void LogErrorWithArgAndValue(const int line, const char* msg,
                             const wchar_t* arg, DWORD error_code) {
  fprintf(stderr,
          "ERROR(" __FILE__ ":%d) error code: %d (0x%08x), argument: %ls: %s\n",
          line, error_code, error_code, arg, msg);
}

bool GetEnv(const wchar_t* name, std::wstring* result) {
  static constexpr size_t kSmallBuf = MAX_PATH;
  WCHAR value[kSmallBuf];
  DWORD size = GetEnvironmentVariableW(name, value, kSmallBuf);
  DWORD err = GetLastError();
  if (size == 0 && err == ERROR_ENVVAR_NOT_FOUND) {
    result->clear();
    return true;
  } else if (0 < size && size < kSmallBuf) {
    *result = value;
    return true;
  } else if (size >= kSmallBuf) {
    std::unique_ptr<WCHAR[]> value_big(new WCHAR[size]);
    GetEnvironmentVariableW(name, value_big.get(), size);
    *result = value_big.get();
    return true;
  } else {
    LogErrorWithArgAndValue(__LINE__, "Failed to read envvar", name, err);
    return false;
  }
}

inline void PrintTestLogStartMarker() {
  // This header marks where --test_output=streamed will start being printed.
  printf(
      "------------------------------------------------------------------------"
      "-----\n");
}

inline bool GetWorkspaceName(std::wstring* result) {
  return GetEnv(L"TEST_WORKSPACE", result) && !result->empty();
}

inline void StripLeadingDotSlash(std::wstring* s) {
  if (s->size() >= 2 && (*s)[0] == L'.' && (*s)[1] == L'/') {
    s->erase(0, 2);
  }
}

bool FindTestBinary(const std::wstring& argv0, std::wstring test_path,
                    std::wstring* result) {
  if (!blaze_util::IsAbsolute(test_path)) {
    std::string argv0_acp;
    uint32_t err;
    if (!blaze_util::WcsToAcp(argv0, &argv0_acp, &err)) {
      LogErrorWithArgAndValue(__LINE__, "Failed to convert string",
                              argv0.c_str(), err);
      return false;
    }

    std::string error;
    std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
        bazel::tools::cpp::runfiles::Runfiles::Create(argv0_acp, &error));
    if (runfiles == nullptr) {
      LogError(__LINE__, "Failed to load runfiles");
      return false;
    }

    std::wstring workspace;
    if (!GetWorkspaceName(&workspace)) {
      LogError(__LINE__, "Failed to read %TEST_WORKSPACE%");
      return false;
    }

    StripLeadingDotSlash(&test_path);
    test_path = workspace + L"/" + test_path;

    std::string utf8_test_path;
    if (!blaze_util::WcsToUtf8(test_path, &utf8_test_path, &err)) {
      LogErrorWithArgAndValue(__LINE__, "Failed to convert string to UTF-8",
                              test_path.c_str(), err);
      return false;
    }

    std::string rloc = runfiles->Rlocation(utf8_test_path);
    if (!blaze_util::Utf8ToWcs(rloc, &test_path, &err)) {
      LogErrorWithArgAndValue(__LINE__, "Failed to convert string",
                              utf8_test_path.c_str(), err);
    }
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
    case WAIT_OBJECT_0: {
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

int wmain(int argc, wchar_t** argv) {
  // TODO(laszlocsomor): Implement the functionality described in
  // https://github.com/laszlocsomor/proposals/blob/win-test-runner/designs/2018-07-18-windows-native-test-runner.md

  const wchar_t* argv0 = argv[0];
  argc--;
  argv++;
  bool suppress_output = false;
  if (argc > 0 && wcscmp(argv[0], L"--no_echo") == 0) {
    // Don't print anything to stdout in this special case.
    // Currently needed for persistent test runner.
    suppress_output = true;
    argc--;
    argv++;
  } else {
    std::wstring test_target;
    if (!GetEnv(L"TEST_TARGET", &test_target)) {
      return 1;
    }
    printf("Executing tests from %ls\n", test_target.c_str());
  }

  if (argc < 1) {
    LogError(__LINE__, "Usage: $0 [--no_echo] <test_path> [test_args...]");
    return 1;
  }

  if (!suppress_output) {
    PrintTestLogStartMarker();
  }

  const wchar_t* test_path_arg = argv[0];
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
