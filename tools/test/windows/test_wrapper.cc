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
#include <lmcons.h>  // UNLEN
#include <windows.h>

#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace {

class Defer {
 public:
  explicit Defer(std::function<void()> f) : f_(f) {}
  ~Defer() { f_(); }

 private:
  std::function<void()> f_;
};

// A lightweight path abstraction that stores a Unicode Windows path.
//
// The class allows extracting the underlying path as a (immutable) string so
// it's easy to pass the path to WinAPI functions, but the class does not allow
// mutating the unterlying path so it's safe to pass around Path objects.
class Path {
 public:
  Path() {}
  Path(const Path& other) : path_(other.path_) {}
  Path(Path&& other) : path_(std::move(other.path_)) {}
  Path& operator=(const Path& other) = delete;
  const std::wstring& Get() const { return path_; }
  bool Set(const std::wstring& path);

  // Makes this path absolute.
  // Returns true if the path was changed (i.e. was not absolute before).
  // Returns false and has no effect if this path was empty or already absolute.
  bool Absolutize(const Path& cwd);

  Path Dirname() const;

 private:
  std::wstring path_;
};

struct UndeclaredOutputs {
  Path root;
  Path zip;
  Path manifest;
  Path annotations;
  Path annotations_dir;
};

void LogError(const int line, const char* msg) {
  printf("ERROR(" __FILE__ ":%d) %s\n", line, msg);
}

void LogErrorWithValue(const int line, const char* msg, DWORD error_code) {
  printf("ERROR(" __FILE__ ":%d) error code: %d (0x%08x): %s\n", line,
         error_code, error_code, msg);
}

void LogErrorWithArgAndValue(const int line, const char* msg, const char* arg,
                             DWORD error_code) {
  printf("ERROR(" __FILE__ ":%d) error code: %d (0x%08x), argument: %s: %s\n",
         line, error_code, error_code, arg, msg);
}

void LogErrorWithArgAndValue(const int line, const char* msg,
                             const wchar_t* arg, DWORD error_code) {
  printf("ERROR(" __FILE__ ":%d) error code: %d (0x%08x), argument: %ls: %s\n",
         line, error_code, error_code, arg, msg);
}

inline bool CreateDirectories(const Path& path) {
  blaze_util::MakeDirectoriesW(bazel::windows::HasUncPrefix(path.Get().c_str())
                                   ? path.Get()
                                   : L"\\\\?\\" + path.Get(),
                               0777);
  return true;
}

inline bool ToInt(const wchar_t* s, int* result) {
  return swscanf_s(s, L"%d", result) == 1;
}

// Converts a Windows-style path to a mixed (Unix-Windows) style.
// The path is mixed-style because it is a Windows path (begins with a drive
// letter) but uses forward slashes as directory separators.
// We must export envvars as mixed style path because some tools confuse the
// backslashes in Windows paths for Unix-style escape characters.
inline std::wstring AsMixedPath(const std::wstring& path) {
  std::wstring value = path;
  std::replace(value.begin(), value.end(), L'\\', L'/');
  return value;
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

bool GetPathEnv(const wchar_t* name, Path* result) {
  std::wstring value;
  if (!GetEnv(name, &value)) {
    return false;
  }
  return result->Set(value);
}

bool SetEnv(const wchar_t* name, const std::wstring& value) {
  if (SetEnvironmentVariableW(name, value.c_str()) != 0) {
    return true;
  } else {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to set envvar", name, err);
    return false;
  }
}

bool UnsetEnv(const wchar_t* name) {
  if (SetEnvironmentVariableW(name, NULL) != 0) {
    return true;
  } else {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to unset envvar", name, err);
    return false;
  }
}

bool GetCwd(Path* result) {
  static constexpr size_t kSmallBuf = MAX_PATH;
  WCHAR value[kSmallBuf];
  DWORD size = GetCurrentDirectoryW(kSmallBuf, value);
  DWORD err = GetLastError();
  if (size > 0 && size < kSmallBuf) {
    return result->Set(value);
  } else if (size >= kSmallBuf) {
    std::unique_ptr<WCHAR[]> value_big(new WCHAR[size]);
    GetCurrentDirectoryW(size, value_big.get());
    return result->Set(value_big.get());
  } else {
    LogErrorWithValue(__LINE__, "Failed to get current directory", err);
    return false;
  }
}

// Set USER as required by the Bazel Test Encyclopedia.
bool ExportUserName() {
  std::wstring value;
  if (!GetEnv(L"USER", &value)) {
    return false;
  }
  if (!value.empty()) {
    // Respect the value passed by Bazel via --test_env.
    return true;
  }
  WCHAR buffer[UNLEN + 1];
  DWORD len = UNLEN + 1;
  if (GetUserNameW(buffer, &len) == 0) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "Failed to query user name", err);
    return false;
  }
  return SetEnv(L"USER", buffer);
}

// Set TEST_SRCDIR as required by the Bazel Test Encyclopedia.
bool ExportSrcPath(const Path& cwd, Path* result) {
  if (!GetPathEnv(L"TEST_SRCDIR", result)) {
    return false;
  }
  return !result->Absolutize(cwd) ||
         SetEnv(L"TEST_SRCDIR", AsMixedPath(result->Get()));
}

// Set TEST_TMPDIR as required by the Bazel Test Encyclopedia.
bool ExportTmpPath(const Path& cwd, Path* result) {
  if (!GetPathEnv(L"TEST_TMPDIR", result) ||
      (result->Absolutize(cwd) &&
       !SetEnv(L"TEST_TMPDIR", AsMixedPath(result->Get())))) {
    return false;
  }
  // Create the test temp directory, which may not exist on the remote host when
  // doing a remote build.
  return CreateDirectories(*result);
}

// Set HOME as required by the Bazel Test Encyclopedia.
bool ExportHome(const Path& test_tmpdir) {
  Path home;
  if (!GetPathEnv(L"HOME", &home)) {
    return false;
  }
  if (blaze_util::IsAbsolute(home.Get())) {
    // Respect the user-defined HOME in case they set passed it with
    // --test_env=HOME or --test_env=HOME=C:\\foo
    return true;
  } else {
    // Set TEST_TMPDIR as required by the Bazel Test Encyclopedia.
    return SetEnv(L"HOME", AsMixedPath(test_tmpdir.Get()));
  }
}

bool ExportRunfiles(const Path& cwd, const Path& test_srcdir) {
  Path runfiles_dir;
  if (!GetPathEnv(L"RUNFILES_DIR", &runfiles_dir) ||
      (runfiles_dir.Absolutize(cwd) &&
       !SetEnv(L"RUNFILES_DIR", AsMixedPath(runfiles_dir.Get())))) {
    return false;
  }

  // TODO(ulfjack): Standardize on RUNFILES_DIR and remove the
  // {JAVA,PYTHON}_RUNFILES vars.
  Path java_rf, py_rf;
  if (!GetPathEnv(L"JAVA_RUNFILES", &java_rf) ||
      (java_rf.Absolutize(cwd) &&
       !SetEnv(L"JAVA_RUNFILES", AsMixedPath(java_rf.Get()))) ||
      !GetPathEnv(L"PYTHON_RUNFILES", &py_rf) ||
      (py_rf.Absolutize(cwd) &&
       !SetEnv(L"PYTHON_RUNFILES", AsMixedPath(py_rf.Get())))) {
    return false;
  }

  std::wstring mf_only_str;
  int mf_only_value = 0;
  if (!GetEnv(L"RUNFILES_MANIFEST_ONLY", &mf_only_str) ||
      (!mf_only_str.empty() && !ToInt(mf_only_str.c_str(), &mf_only_value))) {
    return false;
  }
  if (mf_only_value == 1) {
    // If RUNFILES_MANIFEST_ONLY is set to 1 then test programs should use the
    // manifest file to find their runfiles.
    Path runfiles_mf;
    if (!runfiles_mf.Set(test_srcdir.Get() + L"\\MANIFEST") ||
        !SetEnv(L"RUNFILES_MANIFEST_FILE", AsMixedPath(runfiles_mf.Get()))) {
      return false;
    }
  }

  return true;
}

bool ExportShardStatusFile(const Path& cwd) {
  Path status_file;
  if (!GetPathEnv(L"TEST_SHARD_STATUS_FILE", &status_file) ||
      (!status_file.Get().empty() && status_file.Absolutize(cwd) &&
       !SetEnv(L"TEST_SHARD_STATUS_FILE", status_file.Get()))) {
    return false;
  }

  return status_file.Get().empty() ||
         // The test shard status file is only set for sharded tests.
         CreateDirectories(status_file.Dirname());
}

bool ExportGtestVariables(const Path& test_tmpdir) {
  // # Tell googletest about Bazel sharding.
  std::wstring total_shards_str;
  int total_shards_value = 0;
  if (!GetEnv(L"TEST_TOTAL_SHARDS", &total_shards_str) ||
      (!total_shards_str.empty() &&
       !ToInt(total_shards_str.c_str(), &total_shards_value))) {
    return false;
  }
  if (total_shards_value > 0) {
    std::wstring shard_index;
    if (!GetEnv(L"TEST_SHARD_INDEX", &shard_index) ||
        !SetEnv(L"GTEST_SHARD_INDEX", shard_index) ||
        !SetEnv(L"GTEST_TOTAL_SHARDS", total_shards_str)) {
      return false;
    }
  }
  return SetEnv(L"GTEST_TMP_DIR", AsMixedPath(test_tmpdir.Get()));
}

bool ExportMiscEnvvars(const Path& cwd) {
  for (const wchar_t* name :
       {L"TEST_INFRASTRUCTURE_FAILURE_FILE", L"TEST_LOGSPLITTER_OUTPUT_FILE",
        L"TEST_PREMATURE_EXIT_FILE", L"TEST_UNUSED_RUNFILES_LOG_FILE",
        L"TEST_WARNINGS_OUTPUT_FILE"}) {
    Path value;
    if (!GetPathEnv(name, &value) ||
        (value.Absolutize(cwd) && !SetEnv(name, AsMixedPath(value.Get())))) {
      return false;
    }
  }
  return true;
}

bool GetAndUnexportUndeclaredOutputsEnvvars(const Path& cwd,
                                            UndeclaredOutputs* result) {
  // The test may only see TEST_UNDECLARED_OUTPUTS_DIR and
  // TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR, so keep those but unexport others.
  if (!GetPathEnv(L"TEST_UNDECLARED_OUTPUTS_ZIP", &(result->zip)) ||
      !UnsetEnv(L"TEST_UNDECLARED_OUTPUTS_ZIP") ||

      !GetPathEnv(L"TEST_UNDECLARED_OUTPUTS_MANIFEST", &(result->manifest)) ||
      !UnsetEnv(L"TEST_UNDECLARED_OUTPUTS_MANIFEST") ||

      !GetPathEnv(L"TEST_UNDECLARED_OUTPUTS_ANNOTATIONS",
                  &(result->annotations)) ||
      !UnsetEnv(L"TEST_UNDECLARED_OUTPUTS_ANNOTATIONS") ||

      !GetPathEnv(L"TEST_UNDECLARED_OUTPUTS_DIR", &(result->root)) ||

      !GetPathEnv(L"TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR",
                  &(result->annotations_dir))) {
    return false;
  }

  result->root.Absolutize(cwd);
  result->annotations_dir.Absolutize(cwd);
  result->zip.Absolutize(cwd);
  result->manifest.Absolutize(cwd);
  result->annotations.Absolutize(cwd);

  return SetEnv(L"TEST_UNDECLARED_OUTPUTS_DIR",
                AsMixedPath(result->root.Get())) &&
         SetEnv(L"TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR",
                AsMixedPath(result->annotations_dir.Get())) &&
         CreateDirectories(result->root) &&
         CreateDirectories(result->annotations_dir);
}

inline bool PrintTestLogStartMarker(bool suppress_output) {
  if (suppress_output) {
    return true;
  }

  std::wstring test_target;
  if (!GetEnv(L"TEST_TARGET", &test_target)) {
    return false;
  }
  if (test_target.empty()) {
    // According to the Bazel Test Encyclopedia, setting TEST_TARGET is
    // optional.
    wprintf(L"Executing tests from unknown target\n");
  } else {
    wprintf(L"Executing tests from %s\n", test_target.c_str());
  }

  // This header marks where --test_output=streamed will start being printed.
  printf(
      "------------------------------------------------------------------------"
      "-----\n");
  return true;
}

inline bool GetWorkspaceName(std::wstring* result) {
  return GetEnv(L"TEST_WORKSPACE", result) && !result->empty();
}

inline void StripLeadingDotSlash(std::wstring* s) {
  if (s->size() >= 2 && (*s)[0] == L'.' && (*s)[1] == L'/') {
    s->erase(0, 2);
  }
}

bool FindTestBinary(const Path& argv0, std::wstring test_path, Path* result) {
  if (!blaze_util::IsAbsolute(test_path)) {
    std::string argv0_acp;
    uint32_t err;
    if (!blaze_util::WcsToAcp(argv0.Get(), &argv0_acp, &err)) {
      LogErrorWithArgAndValue(__LINE__, "Failed to convert string",
                              argv0.Get().c_str(), err);
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

  return result->Set(test_path);
}

bool StartSubprocess(const Path& path, HANDLE* process) {
  // kMaxCmdline value: see lpCommandLine parameter of CreateProcessW.
  static constexpr size_t kMaxCmdline = 32768;

  std::unique_ptr<WCHAR[]> cmdline(new WCHAR[kMaxCmdline]);
  size_t len = path.Get().size();
  wcsncpy(cmdline.get(), path.Get().c_str(), len + 1);

  PROCESS_INFORMATION processInfo;
  STARTUPINFOW startupInfo = {0};

  if (CreateProcessW(NULL, cmdline.get(), NULL, NULL, FALSE,
                     CREATE_UNICODE_ENVIRONMENT, NULL, NULL, &startupInfo,
                     &processInfo) != 0) {
    CloseHandle(processInfo.hThread);
    *process = processInfo.hProcess;
    return true;
  } else {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "CreateProcessW failed", err);
    return false;
  }
}

int WaitForSubprocess(HANDLE process) {
  DWORD result = WaitForSingleObject(process, INFINITE);
  switch (result) {
    case WAIT_OBJECT_0: {
      DWORD exit_code;
      if (!GetExitCodeProcess(process, &exit_code)) {
        DWORD err = GetLastError();
        LogErrorWithValue(__LINE__, "GetExitCodeProcess failed", err);
        return 1;
      }
      return exit_code;
    }
    case WAIT_FAILED: {
      DWORD err = GetLastError();
      LogErrorWithValue(__LINE__, "WaitForSingleObject failed", err);
      return 1;
    }
    default:
      LogErrorWithValue(
          __LINE__, "WaitForSingleObject returned unexpected result", result);
      return 1;
  }
}

bool ParseArgs(int argc, wchar_t** argv, Path* out_argv0,
               std::wstring* out_test_path_arg, bool* out_suppress_output) {
  if (!out_argv0->Set(argv[0])) {
    return false;
  }
  argc--;
  argv++;
  *out_suppress_output = false;
  if (argc > 0 && wcscmp(argv[0], L"--no_echo") == 0) {
    // Don't print anything to stdout in this special case.
    // Currently needed for persistent test runner.
    *out_suppress_output = true;
    argc--;
    argv++;
  }

  if (argc < 1) {
    LogError(__LINE__, "Usage: $0 [--no_echo] <test_path> [test_args...]");
    return false;
  }

  *out_test_path_arg = argv[0];
  return true;
}

int RunSubprocess(const Path& test_path) {
  HANDLE process;
  if (!StartSubprocess(test_path, &process)) {
    return 1;
  }
  Defer close_process([process]() { CloseHandle(process); });

  return WaitForSubprocess(process);
}

bool Path::Set(const std::wstring& path) {
  std::wstring result;
  std::string error;
  if (!blaze_util::AsWindowsPath(path, &result, &error)) {
    LogError(__LINE__, error.c_str());
    return false;
  }
  path_ = result;
  return true;
}

bool Path::Absolutize(const Path& cwd) {
  if (!path_.empty() && !blaze_util::IsAbsolute(path_)) {
    path_ = cwd.path_ + L"\\" + path_;
    return true;
  } else {
    return false;
  }
}

Path Path::Dirname() const {
  Path result;
  result.path_ = blaze_util::SplitPathW(path_).first;
  return result;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  Path argv0;
  std::wstring test_path_arg;
  bool suppress_output = false;
  Path test_path, exec_root, srcdir, tmpdir, xml_output;
  UndeclaredOutputs undecl;
  if (!ParseArgs(argc, argv, &argv0, &test_path_arg, &suppress_output) ||
      !PrintTestLogStartMarker(suppress_output) ||
      !FindTestBinary(argv0, test_path_arg, &test_path) ||
      !GetCwd(&exec_root) || !ExportUserName() ||
      !ExportSrcPath(exec_root, &srcdir) ||
      !ExportTmpPath(exec_root, &tmpdir) || !ExportHome(tmpdir) ||
      !ExportRunfiles(exec_root, srcdir) || !ExportShardStatusFile(exec_root) ||
      !ExportGtestVariables(tmpdir) || !ExportMiscEnvvars(exec_root) ||
      !GetAndUnexportUndeclaredOutputsEnvvars(exec_root, &undecl)) {
    return 1;
  }

  return RunSubprocess(test_path);
}
