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
#include <vector>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/platform_utils.h"
#include "third_party/ijar/zip.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "tools/test/windows/tw.h"

namespace bazel {
namespace tools {
namespace test_wrapper {
namespace {

class Defer {
 public:
  explicit Defer(std::function<void()> f) : f_(f) {}
  ~Defer() { f_(); }

  void DoNow() {
    f_();
    f_ = kEmpty;
  }

 private:
  std::function<void()> f_;
  static const std::function<void()> kEmpty;
};

const std::function<void()> Defer::kEmpty = []() {};

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

void LogError(const int line, const wchar_t* msg) {
#define _WSTR_HELPER_1(x) L##x
#define _WSTR_HELPER_2(x) _WSTR_HELPER_1(x)
  wprintf(L"ERROR(" _WSTR_HELPER_2(__FILE__) L":%d) %s\n", line, msg);
#undef _WSTR_HELPER_2
#undef _WSTR_HELPER_1
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

bool WcsToAcp(const std::wstring& wcs, std::string* acp) {
  uint32_t err;
  if (!blaze_util::WcsToAcp(wcs, acp, &err)) {
    LogErrorWithArgAndValue(__LINE__, "Failed to convert string", wcs.c_str(),
                            err);
    return false;
  }
  return true;
}

// Converts a Windows-style path to a mixed (Unix-Windows) style.
// The path is mixed-style because it is a Windows path (begins with a drive
// letter) but uses forward slashes as directory separators.
// We must export envvars as mixed style path because some tools confuse the
// backslashes in Windows paths for Unix-style escape characters.
std::wstring AsMixedPath(const std::wstring& path) {
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

bool SetPathEnv(const wchar_t* name, const Path& path) {
  return SetEnv(name, AsMixedPath(path.Get()));
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
  return !result->Absolutize(cwd) || SetPathEnv(L"TEST_SRCDIR", *result);
}

// Set TEST_TMPDIR as required by the Bazel Test Encyclopedia.
bool ExportTmpPath(const Path& cwd, Path* result) {
  if (!GetPathEnv(L"TEST_TMPDIR", result) ||
      (result->Absolutize(cwd) && !SetPathEnv(L"TEST_TMPDIR", *result))) {
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
    return SetPathEnv(L"HOME", test_tmpdir);
  }
}

bool ExportRunfiles(const Path& cwd, const Path& test_srcdir) {
  Path runfiles_dir;
  if (!GetPathEnv(L"RUNFILES_DIR", &runfiles_dir) ||
      (runfiles_dir.Absolutize(cwd) &&
       !SetPathEnv(L"RUNFILES_DIR", runfiles_dir))) {
    return false;
  }

  // TODO(ulfjack): Standardize on RUNFILES_DIR and remove the
  // {JAVA,PYTHON}_RUNFILES vars.
  Path java_rf, py_rf;
  if (!GetPathEnv(L"JAVA_RUNFILES", &java_rf) ||
      (java_rf.Absolutize(cwd) && !SetPathEnv(L"JAVA_RUNFILES", java_rf)) ||
      !GetPathEnv(L"PYTHON_RUNFILES", &py_rf) ||
      (py_rf.Absolutize(cwd) && !SetPathEnv(L"PYTHON_RUNFILES", py_rf))) {
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
        !SetPathEnv(L"RUNFILES_MANIFEST_FILE", runfiles_mf)) {
      return false;
    }
  }

  return true;
}

bool ExportShardStatusFile(const Path& cwd) {
  Path status_file;
  if (!GetPathEnv(L"TEST_SHARD_STATUS_FILE", &status_file) ||
      (!status_file.Get().empty() && status_file.Absolutize(cwd) &&
       !SetPathEnv(L"TEST_SHARD_STATUS_FILE", status_file))) {
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
  return SetPathEnv(L"GTEST_TMP_DIR", test_tmpdir);
}

bool ExportMiscEnvvars(const Path& cwd) {
  for (const wchar_t* name :
       {L"TEST_INFRASTRUCTURE_FAILURE_FILE", L"TEST_LOGSPLITTER_OUTPUT_FILE",
        L"TEST_PREMATURE_EXIT_FILE", L"TEST_UNUSED_RUNFILES_LOG_FILE",
        L"TEST_WARNINGS_OUTPUT_FILE"}) {
    Path value;
    if (!GetPathEnv(name, &value) ||
        (value.Absolutize(cwd) && !SetPathEnv(name, value))) {
      return false;
    }
  }
  return true;
}

bool _GetFileListRelativeTo(
    const std::wstring& unc_root, const std::wstring& subdir,
    std::vector<bazel::tools::test_wrapper::FileInfo>* result) {
  const std::wstring full_subdir =
      unc_root + (subdir.empty() ? L"" : (L"\\" + subdir)) + L"\\*";
  WIN32_FIND_DATAW info;
  HANDLE handle = FindFirstFileW(full_subdir.c_str(), &info);
  if (handle == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    if (err == ERROR_FILE_NOT_FOUND) {
      // No files found, nothing to do.
      return true;
    }
    LogErrorWithArgAndValue(__LINE__, "Failed to list directory contents",
                            full_subdir.c_str(), err);
    return false;
  }

  Defer close_handle([handle]() { FindClose(handle); });
  static const std::wstring kDot(1, L'.');
  static const std::wstring kDotDot(2, L'.');
  std::vector<std::wstring> subdirectories;
  while (true) {
    if (kDot != info.cFileName && kDotDot != info.cFileName) {
      std::wstring rel_path =
          subdir.empty() ? info.cFileName : (subdir + L"\\" + info.cFileName);
      if (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        subdirectories.push_back(rel_path);
      } else {
        if (info.nFileSizeHigh > 0 || info.nFileSizeLow > INT_MAX) {
          // devtools_ijar::Stat::total_size is declared as `int`, so the file
          // size limit is INT_MAX. Additionally we limit the files to be below
          // 4 GiB, not only because int is typically 4 bytes long, but also
          // because such huge files are unreasonably large as an undeclared
          // output.
          LogErrorWithArgAndValue(__LINE__, "File is too large to archive",
                                  rel_path.c_str(), 0);
          return false;
        }

        result->push_back({rel_path,
                           // File size is already validated to be smaller than
                           // min(INT_MAX, 4 GiB)
                           static_cast<int>(info.nFileSizeLow)});
      }
    }
    if (FindNextFileW(handle, &info) == 0) {
      DWORD err = GetLastError();
      if (err == ERROR_NO_MORE_FILES) {
        break;
      }
      LogErrorWithArgAndValue(__LINE__,
                              "Failed to get next element in directory",
                              (unc_root + L"\\" + subdir).c_str(), err);
      return false;
    }
  }
  close_handle.DoNow();

  for (const auto& s : subdirectories) {
    if (!_GetFileListRelativeTo(unc_root, s, result)) {
      return false;
    }
  }
  return true;
}

bool GetFileListRelativeTo(
    const Path& root,
    std::vector<bazel::tools::test_wrapper::FileInfo>* result) {
  if (!blaze_util::IsAbsolute(root.Get())) {
    LogError(__LINE__, "Root should be absolute");
    return false;
  }

  return _GetFileListRelativeTo(bazel::windows::HasUncPrefix(root.Get().c_str())
                                    ? root.Get()
                                    : L"\\\\?\\" + root.Get(),
                                std::wstring(), result);
}

bool ToZipEntryPaths(
    const Path& root,
    const std::vector<bazel::tools::test_wrapper::FileInfo>& files,
    ZipEntryPaths* result) {
  std::string acp_root;
  if (!WcsToAcp(AsMixedPath(bazel::windows::HasUncPrefix(root.Get().c_str())
                                ? root.Get().substr(4)
                                : root.Get()),
                &acp_root)) {
    LogError(__LINE__,
             (std::wstring(L"Failed to convert path \"") + root.Get() + L"\"")
                 .c_str());
    return false;
  }

  // Convert all UTF-16 paths to ANSI paths.
  std::vector<std::string> acp_file_list;
  acp_file_list.reserve(files.size());
  for (const auto& e : files) {
    std::string acp_path;
    if (!WcsToAcp(AsMixedPath(e.rel_path), &acp_path)) {
      LogError(__LINE__,
               (std::wstring(L"Failed to convert path ") + e.rel_path + L"\"")
                   .c_str());
      return false;
    }
    acp_file_list.push_back(acp_path);
  }

  result->Create(acp_root, acp_file_list);
  return true;
}

bool CreateZipBuilder(const Path& zip, const ZipEntryPaths& entry_paths,
                      std::unique_ptr<devtools_ijar::ZipBuilder>* result) {
  const devtools_ijar::u8 estimated_size =
      devtools_ijar::ZipBuilder::EstimateSize(entry_paths.AbsPathPtrs(),
                                              entry_paths.EntryPathPtrs(),
                                              entry_paths.Size());

  if (estimated_size == 0) {
    LogError(__LINE__, "Failed to estimate zip size");
    return false;
  }

  std::string acp_zip;
  if (!WcsToAcp(zip.Get(), &acp_zip)) {
    LogError(
        __LINE__,
        (std::wstring(L"Failed to convert path ") + zip.Get() + L"\"").c_str());
    return false;
  }

  result->reset(
      devtools_ijar::ZipBuilder::Create(acp_zip.c_str(), estimated_size));
  if (result->get() == nullptr) {
    LogErrorWithValue(__LINE__, "Failed to create zip builder", errno);
    return false;
  }
  return true;
}

bool OpenFileForWriting(const std::wstring& path, HANDLE* result) {
  *result = CreateFileW(bazel::windows::HasUncPrefix(path.c_str())
                            ? path.c_str()
                            : (L"\\\\?\\" + path).c_str(),
                        GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE,
                        NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (*result == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to open file", path.c_str(), err);
    return false;
  }
  return true;
}

bool OpenExistingFileForRead(const Path& abs_path, HANDLE* result) {
  *result = CreateFileW(bazel::windows::HasUncPrefix(abs_path.Get().c_str())
                            ? abs_path.Get().c_str()
                            : (L"\\\\?\\" + abs_path.Get()).c_str(),
                        GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (*result == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to open file",
                            abs_path.Get().c_str(), err);
    return false;
  }
  return true;
}

bool TouchFile(const std::wstring& path) {
  HANDLE handle;
  if (!OpenFileForWriting(path, &handle)) {
    return false;
  }
  CloseHandle(handle);
  return true;
}

bool ReadCompleteFile(HANDLE handle, uint8_t* dest, DWORD max_read) {
  if (max_read == 0) {
    return true;
  }

  DWORD total_read = 0;
  DWORD read = 0;
  do {
    if (!ReadFile(handle, dest + total_read, max_read - total_read, &read,
                  NULL)) {
      DWORD err = GetLastError();
      LogErrorWithValue(__LINE__, "Failed to read file", err);
      return false;
    }
    total_read += read;
  } while (read > 0 && total_read < max_read);
  return true;
}

bool ExportXmlPath(const Path& cwd) {
  Path result;
  if (!GetPathEnv(L"XML_OUTPUT_FILE", &result)) {
    return false;
  }
  result.Absolutize(cwd);
  std::wstring unix_result = AsMixedPath(result.Get());
  return SetEnv(L"XML_OUTPUT_FILE", unix_result) &&
         // TODO(ulfjack): Update Gunit to accept XML_OUTPUT_FILE and drop the
         // GUNIT_OUTPUT env variable.
         SetEnv(L"GUNIT_OUTPUT", L"xml:" + unix_result) &&
         CreateDirectories(result.Dirname()) &&
         TouchFile(result.Get() + L".log");
}

devtools_ijar::u4 GetZipAttr(const FileInfo& info) {
  devtools_ijar::Stat file_stat;
  file_stat.total_size = info.size;
  file_stat.is_directory = false;
  // Set 0777 permission mask inside the zip for sake of simplicity.
  file_stat.file_mode = S_IFREG | 0777;
  return devtools_ijar::stat_to_zipattr(file_stat);
}

bool GetZipEntryPtr(devtools_ijar::ZipBuilder* zip_builder,
                    const char* entry_name, const devtools_ijar::u4 attr,
                    devtools_ijar::u1** result) {
  *result = zip_builder->NewFile(entry_name, attr);
  if (*result == nullptr) {
    LogError(__LINE__, (std::string("Failed to add new zip entry for file \"") +
                        entry_name + "\": " + zip_builder->GetError())
                           .c_str());
    return false;
  }
  return true;
}

bool CreateZip(const Path& root, const std::vector<FileInfo>& files,
               const Path& abs_zip) {
  bool restore_oem_api = false;
  if (!AreFileApisANSI()) {
    // devtools_ijar::ZipBuilder uses the ANSI file APIs so we must set the
    // active code page to ANSI.
    SetFileApisToANSI();
    restore_oem_api = true;
  }
  Defer restore_file_apis([restore_oem_api]() {
    if (restore_oem_api) {
      SetFileApisToOEM();
    }
  });

  ZipEntryPaths zip_entry_paths;
  if (!ToZipEntryPaths(root, files, &zip_entry_paths)) {
    LogError(__LINE__, "Failed to create zip entry paths");
    return false;
  }

  std::unique_ptr<devtools_ijar::ZipBuilder> zip_builder;
  if (!CreateZipBuilder(abs_zip, zip_entry_paths, &zip_builder)) {
    LogError(__LINE__, "Failed to create zip builder");
    return false;
  }

  for (size_t i = 0; i < files.size(); ++i) {
    HANDLE handle;
    Path path;
    if (!path.Set(root.Get() + L"\\" + files[i].rel_path) ||
        !OpenExistingFileForRead(path, &handle)) {
      LogError(__LINE__,
               (std::wstring(L"Failed to open file \"") + path.Get() + L"\"")
                   .c_str());
      return false;
    }
    Defer close_file([handle]() { CloseHandle(handle); });
    devtools_ijar::u1* dest;
    if (!GetZipEntryPtr(zip_builder.get(), zip_entry_paths.EntryPathPtrs()[i],
                        GetZipAttr(files[i]), &dest) ||
        !ReadCompleteFile(handle, dest, files[i].size)) {
      LogError(__LINE__, (std::wstring(L"Failed to dump file \"") + path.Get() +
                          +L"\" into zip")
                             .c_str());
      return false;
    }

    if (zip_builder->FinishFile(files[i].size, /* compress */ false,
                                /* compute_crc */ true) == -1) {
      LogError(__LINE__, (std::wstring(L"Failed to finish writing file \"") +
                          path.Get() + L"\" to zip")
                             .c_str());
      return false;
    }
  }

  if (zip_builder->Finish() == -1) {
    LogError(__LINE__, (std::string("Failed to add file to zip: ") +
                        zip_builder->GetError())
                           .c_str());
    return false;
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

  return SetPathEnv(L"TEST_UNDECLARED_OUTPUTS_DIR", result->root) &&
         SetPathEnv(L"TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR",
                    result->annotations_dir) &&
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
    if (!WcsToAcp(argv0.Get(), &argv0_acp)) {
      LogError(__LINE__,
               (std::wstring(L"Failed to convert path ") + argv0.Get() + L"\"")
                   .c_str());
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
    uint32_t err;
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

bool AddCommandLineArg(const wchar_t* arg, const size_t arg_size,
                       const bool first, wchar_t* cmdline,
                       const size_t cmdline_limit, size_t* inout_cmdline_len) {
  if (arg_size == 0) {
    const size_t len = (first ? 0 : 1) + 2;
    if (*inout_cmdline_len + len >= cmdline_limit) {
      LogError(__LINE__,
               (std::wstring(L"Failed to add command line argument \"") + arg +
                L"\"; command would be too long")
                   .c_str());
      return false;
    }

    size_t offset = *inout_cmdline_len;
    if (!first) {
      cmdline[offset] = L' ';
      offset += 1;
    }
    cmdline[offset] = L'"';
    cmdline[offset + 1] = L'"';
    *inout_cmdline_len += len;
    return true;
  } else {
    const size_t len = (first ? 0 : 1) + arg_size;
    if (*inout_cmdline_len + len >= cmdline_limit) {
      LogError(__LINE__,
               (std::wstring(L"Failed to add command line argument \"") + arg +
                L"\"; command would be too long")
                   .c_str());
      return false;
    }

    size_t offset = *inout_cmdline_len;
    if (!first) {
      cmdline[offset] = L' ';
      offset += 1;
    }
    wcsncpy(cmdline + offset, arg, arg_size);
    offset += arg_size;
    *inout_cmdline_len += len;
    return true;
  }
}

bool CreateCommandLine(const Path& path,
                       const std::vector<const wchar_t*>& args,
                       std::unique_ptr<WCHAR[]>* result) {
  // kMaxCmdline value: see lpCommandLine parameter of CreateProcessW.
  static constexpr size_t kMaxCmdline = 32767;

  // Add an extra character for the final null-terminator.
  result->reset(new WCHAR[kMaxCmdline + 1]);

  size_t total_len = 0;
  if (!AddCommandLineArg(path.Get().c_str(), path.Get().size(), true,
                         result->get(), kMaxCmdline, &total_len)) {
    return false;
  }

  for (const auto arg : args) {
    if (!AddCommandLineArg(arg, wcslen(arg), false, result->get(), kMaxCmdline,
                           &total_len)) {
      return false;
    }
  }
  // Add final null-terminator. There's surely enough room for it:
  // AddCommandLineArg kept validating that we stay under the limit of
  // kMaxCmdline, and the buffer is one WCHAR larger than that.
  result->get()[total_len] = 0;
  return true;
}

bool StartSubprocess(const Path& path, const std::vector<const wchar_t*>& args,
                     HANDLE* process) {
  std::unique_ptr<WCHAR[]> cmdline;
  if (!CreateCommandLine(path, args, &cmdline)) {
    return false;
  }
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
               std::wstring* out_test_path_arg, bool* out_suppress_output,
               std::vector<const wchar_t*>* out_args) {
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
  out_args->clear();
  out_args->reserve(argc - 1);
  for (int i = 1; i < argc; i++) {
    out_args->push_back(argv[i]);
  }
  return true;
}

int RunSubprocess(const Path& test_path,
                  const std::vector<const wchar_t*>& args) {
  HANDLE process;
  if (!StartSubprocess(test_path, args, &process)) {
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

void ZipEntryPaths::Create(const std::string& root,
                           const std::vector<std::string>& relative_paths) {
  size_ = relative_paths.size();

  size_t total_size = 0;
  for (const auto& e : relative_paths) {
    // Increase total size for absolute paths by <root> + "/" + <path> +
    // null-terminator.
    total_size += root.size() + 1 + e.size() + 1;
  }

  // Store all absolute paths in one continuous char array.
  abs_paths_.reset(new char[total_size]);

  // Store pointers in two arrays. The pointers point into `abs_path`.
  // We'll pass these paths to devtools_ijar::ZipBuilder::EstimateSize that
  // expects an array of char pointers. The last element must be NULL, so
  // allocate one extra pointer.
  abs_path_ptrs_.reset(new char*[relative_paths.size() + 1]);
  entry_path_ptrs_.reset(new char*[relative_paths.size() + 1]);

  char* p = abs_paths_.get();
  // Create all full paths (root + '/' + relative_paths[i] + '\0').
  //
  // If `root` is "c:/foo", then store the following:
  //
  // - Store each absolute path consecutively in `abs_paths_` (via `p`).
  //   Store paths with forward slashes and not backslashes, because we use them
  //   as zip entry paths, as well as paths we open with CreateFileA (which can
  //   convert these paths internally to Windows-style).
  //   Example: "c:/foo/bar.txt\0c:/foo/sub/baz.txt\0"
  //
  // - Store pointers in `abs_path_ptrs_`, pointing to the start of each
  //   string inside `abs_paths_`.
  //   Example: "c:/foo/bar.txt\0c:/foo/sub/baz.txt\0"
  //             ^ here          ^ here
  //
  // - Store pointers in `entry_path_ptrs_`, pointing to the start of each
  //   zip entry path inside `abs_paths_`, which is the part of each path
  //   that's relative to `root`.
  //   Example: "c:/foo/bar.txt\0c:/foo/sub/baz.txt\0"
  //                    ^ here          ^ here
  //
  // - Because the ZipBuilder requires that the file paths and zip entry paths
  //   are null-terminated arrays, we insert an extra null at their ends.
  for (size_t i = 0; i < relative_paths.size(); ++i) {
    abs_path_ptrs_.get()[i] = p;
    strncpy(p, root.c_str(), root.size());
    p += root.size();
    *p++ = '/';
    entry_path_ptrs_.get()[i] = p;
    strncpy(p, relative_paths[i].c_str(), relative_paths[i].size() + 1);
    p += relative_paths[i].size() + 1;
  }
  abs_path_ptrs_.get()[relative_paths.size()] = nullptr;
  entry_path_ptrs_.get()[relative_paths.size()] = nullptr;
}

int Main(int argc, wchar_t** argv) {
  Path argv0;
  std::wstring test_path_arg;
  bool suppress_output = false;
  Path test_path, exec_root, srcdir, tmpdir, xml_output;
  UndeclaredOutputs undecl;
  std::vector<const wchar_t*> args;
  if (!ParseArgs(argc, argv, &argv0, &test_path_arg, &suppress_output, &args) ||
      !PrintTestLogStartMarker(suppress_output) ||
      !FindTestBinary(argv0, test_path_arg, &test_path) ||
      !GetCwd(&exec_root) || !ExportUserName() ||
      !ExportSrcPath(exec_root, &srcdir) ||
      !ExportTmpPath(exec_root, &tmpdir) || !ExportHome(tmpdir) ||
      !ExportRunfiles(exec_root, srcdir) || !ExportShardStatusFile(exec_root) ||
      !ExportGtestVariables(tmpdir) || !ExportMiscEnvvars(exec_root) ||
      !ExportXmlPath(exec_root) ||
      !GetAndUnexportUndeclaredOutputsEnvvars(exec_root, &undecl)) {
    return 1;
  }

  return RunSubprocess(test_path, args);
}

namespace testing {

bool TestOnly_GetEnv(const wchar_t* name, std::wstring* result) {
  return GetEnv(name, result);
}

bool TestOnly_GetFileListRelativeTo(const std::wstring& abs_root,
                                    std::vector<FileInfo>* result) {
  Path root;
  return blaze_util::IsAbsolute(abs_root) && root.Set(abs_root) &&
         GetFileListRelativeTo(root, result);
}

bool TestOnly_ToZipEntryPaths(
    const std::wstring& abs_root,
    const std::vector<bazel::tools::test_wrapper::FileInfo>& files,
    ZipEntryPaths* result) {
  Path root;
  return blaze_util::IsAbsolute(abs_root) && root.Set(abs_root) &&
         ToZipEntryPaths(root, files, result);
}

bool TestOnly_CreateZip(const std::wstring& abs_root,
                        const std::vector<FileInfo>& files,
                        const std::wstring& abs_zip) {
  Path root, zip;
  return blaze_util::IsAbsolute(abs_root) && root.Set(abs_root) &&
         blaze_util::IsAbsolute(abs_zip) && zip.Set(abs_zip) &&
         CreateZip(root, files, zip);
}

bool TestOnly_AsMixedPath(const std::wstring& path, std::string* result) {
  return WcsToAcp(
      AsMixedPath(bazel::windows::HasUncPrefix(path.c_str()) ? path.substr(4)
                                                             : path),
      result);
}

}  // namespace testing
}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
