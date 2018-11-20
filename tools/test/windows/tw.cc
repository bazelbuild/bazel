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

#include "tools/test/windows/tw.h"

#define WIN32_LEAN_AND_MEAN
#include <lmcons.h>  // UNLEN
#include <windows.h>

#include <errno.h>
#include <limits.h>  // INT_MAX
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <wchar.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"
#include "third_party/ijar/common.h"
#include "third_party/ijar/platform_utils.h"
#include "third_party/ijar/zip.h"
#include "tools/cpp/runfiles/runfiles.h"

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

// Streams data from an input to two outputs.
// Inspired by tee(1) in the GNU coreutils.
class TeeImpl : Tee {
 public:
  // Creates a background thread to stream data from `input` to the two outputs.
  // The thread terminates when ReadFile fails on the input (e.g. the input is
  // the reading end of a pipe and the writing end is closed) or when WriteFile
  // fails on one of the outputs (e.g. the same output handle is closed
  // elsewhere).
  static bool Create(bazel::windows::AutoHandle* input,
                     bazel::windows::AutoHandle* output1,
                     bazel::windows::AutoHandle* output2,
                     std::unique_ptr<Tee>* result);

 private:
  static DWORD WINAPI ThreadFunc(LPVOID lpParam);

  TeeImpl(bazel::windows::AutoHandle* input,
          bazel::windows::AutoHandle* output1,
          bazel::windows::AutoHandle* output2)
      : input_(input), output1_(output1), output2_(output2) {}
  TeeImpl(const TeeImpl&) = delete;
  TeeImpl& operator=(const TeeImpl&) = delete;

  bool MainFunc() const;

  bazel::windows::AutoHandle input_;
  bazel::windows::AutoHandle output1_;
  bazel::windows::AutoHandle output2_;
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

void LogError(const int line) { printf("ERROR(" __FILE__ ":%d)\n", line); }

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

std::wstring AddUncPrefixMaybe(const Path& p) {
  return bazel::windows::HasUncPrefix(p.Get().c_str())
      ? p.Get()
      : (std::wstring(L"\\\\?\\") + p.Get());
}

std::wstring RemoveUncPrefixMaybe(const Path& p) {
  return bazel::windows::HasUncPrefix(p.Get().c_str())
      ? p.Get().substr(4)
      : p.Get();
}

inline bool CreateDirectories(const Path& path) {
  blaze_util::MakeDirectoriesW(AddUncPrefixMaybe(path), 0777);
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
    LogError(__LINE__, name);
    return false;
  }
  return result->Set(value);
}

bool GetIntEnv(const wchar_t* name, std::wstring* as_wstr, int* as_int) {
  *as_int = 0;
  if (!GetEnv(name, as_wstr) ||
      (!as_wstr->empty() && !ToInt(as_wstr->c_str(), as_int))) {
    LogError(__LINE__, name);
    return false;
  }
  return true;
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
  if (!GetIntEnv(L"RUNFILES_MANIFEST_ONLY", &mf_only_str, &mf_only_value)) {
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
  if (!GetIntEnv(L"TEST_TOTAL_SHARDS", &total_shards_str,
                 &total_shards_value)) {
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

bool _GetFileListRelativeTo(const std::wstring& unc_root,
                            const std::wstring& subdir, int depth_limit,
                            std::vector<FileInfo>* result) {
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
        if (depth_limit != 0) {
          // depth_limit is negative ==> unlimited depth
          // depth_limit is zero     ==> do not recurse further
          // depth_limit is positive ==> recurse further
          subdirectories.push_back(rel_path);
        }
        result->push_back(FileInfo(rel_path));
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

        result->push_back(FileInfo(rel_path,
                                   // File size is already validated to be
                                   // smaller than min(INT_MAX, 4 GiB)
                                   static_cast<int>(info.nFileSizeLow)));
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

  if (depth_limit != 0) {
    // depth_limit is negative ==> unlimited depth
    // depth_limit is zero     ==> do not recurse further
    // depth_limit is positive ==> recurse further
    for (const auto& s : subdirectories) {
      if (!_GetFileListRelativeTo(
              unc_root, s, depth_limit > 0 ? depth_limit - 1 : depth_limit,
              result)) {
        return false;
      }
    }
  }
  return true;
}

bool GetFileListRelativeTo(const Path& root, std::vector<FileInfo>* result,
                           int depth_limit = -1) {
  if (!blaze_util::IsAbsolute(root.Get())) {
    LogError(__LINE__, "Root should be absolute");
    return false;
  }

  return _GetFileListRelativeTo(AddUncPrefixMaybe(root), std::wstring(),
                                depth_limit, result);
}

bool ToZipEntryPaths(const Path& root, const std::vector<FileInfo>& files,
                     ZipEntryPaths* result) {
  std::string acp_root;
  if (!WcsToAcp(AsMixedPath(RemoveUncPrefixMaybe(root)), &acp_root)) {
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
    if (!WcsToAcp(AsMixedPath(e.RelativePath()), &acp_path)) {
      LogError(__LINE__, (std::wstring(L"Failed to convert path \"") +
                          e.RelativePath() + L"\"")
                             .c_str());
      return false;
    }
    if (e.IsDirectory()) {
      acp_path += "/";
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
    LogError(__LINE__,
             (std::wstring(L"Failed to convert path \"") + zip.Get() + L"\"")
                 .c_str());
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

bool OpenFileForWriting(const Path& path, bazel::windows::AutoHandle* result) {
  HANDLE h = CreateFileW(AddUncPrefixMaybe(path).c_str(), GENERIC_WRITE,
                         FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                         CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to open file", path.Get().c_str(),
                            err);
    return false;
  }
  *result = h;
  return true;
}

bool OpenExistingFileForRead(const Path& abs_path,
                             bazel::windows::AutoHandle* result) {
  HANDLE h = CreateFileW(AddUncPrefixMaybe(abs_path).c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to open file",
                            abs_path.Get().c_str(), err);
    return false;
  }
  *result = h;
  return true;
}

bool TouchFile(const Path& path) {
  bazel::windows::AutoHandle handle;
  return OpenFileForWriting(path, &handle);
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

bool WriteToFile(HANDLE output, const void* buffer, const size_t size) {
  // Write `size` many bytes to the output file.
  DWORD total_written = 0;
  while (total_written < size) {
    DWORD written;
    if (!WriteFile(output, static_cast<const uint8_t*>(buffer) + total_written,
                   size - total_written, &written, NULL)) {
      DWORD err = GetLastError();
      LogErrorWithValue(__LINE__, "Failed to write file", err);
      return false;
    }
    total_written += written;
  }
  return true;
}

bool AppendFileTo(const Path& file, const size_t total_size, HANDLE output) {
  bazel::windows::AutoHandle input;
  if (!OpenExistingFileForRead(file, &input)) {
    LogError(
        __LINE__,
        (std::wstring(L"Failed to open file \"") + file.Get() + L"\"").c_str());
    return false;
  }

  const size_t buf_size = std::min<size_t>(total_size, /* 10 MB */ 10000000);
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[buf_size]);

  while (true) {
    // Read at most `buf_size` many bytes from the input file.
    DWORD read = 0;
    if (!ReadFile(input, buffer.get(), buf_size, &read, NULL)) {
      DWORD err = GetLastError();
      LogErrorWithArgAndValue(__LINE__, "Failed to read file",
                              file.Get().c_str(), err);
      return false;
    }
    if (read == 0) {
      // Reached end of input file.
      return true;
    }
    if (!WriteToFile(output, buffer.get(), read)) {
      LogError(__LINE__, (std::wstring(L"Failed to append file \"") +
                          file.Get().c_str() + L"\"")
                             .c_str());
      return false;
    }
  }
  return true;
}

// Returns the MIME type of the file name.
// If the MIME type is unknown or an error occurs, the method returns
// "application/octet-stream".
std::string GetMimeType(const std::string& filename) {
  static constexpr char* kDefaultMimeType = "application/octet-stream";
  std::string::size_type pos = filename.find_last_of('.');
  if (pos == std::string::npos) {
    return kDefaultMimeType;
  }
  char data[1000];
  DWORD data_size = 1000 * sizeof(char);
  if (RegGetValueA(HKEY_CLASSES_ROOT, filename.c_str() + pos, "Content Type",
                   RRF_RT_REG_SZ, NULL, data, &data_size) == ERROR_SUCCESS) {
    return data;
  }
  // The file extension is unknown, or it does not have a "Content Type" value,
  // or the value is too long. We don't care; just return the default.
  return kDefaultMimeType;
}

bool CreateUndeclaredOutputsManifestContent(const std::vector<FileInfo>& files,
                                            std::string* result) {
  std::stringstream stm;
  for (const auto& e : files) {
    if (!e.IsDirectory()) {
      // For each file, write a tab-separated line to the manifest with name
      // (relative to TEST_UNDECLARED_OUTPUTS_DIR), size, and mime type.
      // Example:
      //   foo.txt<TAB>9<TAB>text/plain
      //   bar/baz<TAB>2944<TAB>application/octet-stream
      std::string acp_path;
      if (!WcsToAcp(AsMixedPath(e.RelativePath()), &acp_path)) {
        return false;
      }

      stm << acp_path << "\t" << e.Size() << "\t" << GetMimeType(acp_path)
          << "\n";
    }
  }
  *result = stm.str();
  return true;
}

bool CreateUndeclaredOutputsManifest(const std::vector<FileInfo>& files,
                                     const Path& output) {
  std::string content;
  if (!CreateUndeclaredOutputsManifestContent(files, &content)) {
    LogError(__LINE__,
             (std::wstring(L"Failed to create manifest content for file \"") +
              output.Get() + L"\"")
                 .c_str());
    return false;
  }

  bazel::windows::AutoHandle handle;
  if (!OpenFileForWriting(output, &handle)) {
    LogError(__LINE__, (std::wstring(L"Failed to open file for writing \"") +
                        output.Get() + L"\"")
                           .c_str());
    return false;
  }

  if (!WriteToFile(handle, content.c_str(), content.size())) {
    LogError(__LINE__,
             (std::wstring(L"Failed to write file \"") + output.Get() + L"\"")
                 .c_str());
    return false;
  }
  return true;
}

bool ExportXmlPath(const Path& cwd, Path* test_outerr) {
  Path xml_log;
  if (!GetPathEnv(L"XML_OUTPUT_FILE", &xml_log)) {
    LogError(__LINE__);
    return false;
  }
  xml_log.Absolutize(cwd);
  if (!test_outerr->Set(xml_log.Get() + L".log")) {
    LogError(__LINE__);
    return false;
  }
  std::wstring unix_result = AsMixedPath(xml_log.Get());
  return SetEnv(L"XML_OUTPUT_FILE", unix_result) &&
         // TODO(ulfjack): Update Gunit to accept XML_OUTPUT_FILE and drop the
         // GUNIT_OUTPUT env variable.
         SetEnv(L"GUNIT_OUTPUT", L"xml:" + unix_result) &&
         CreateDirectories(xml_log.Dirname()) && TouchFile(*test_outerr);
}

devtools_ijar::u4 GetZipAttr(const FileInfo& info) {
  // We use these hard-coded Unix permission masks because they are:
  // - stable, so the zip file is deterministic
  // - useful, because stat_to_zipattr expects a mode_t
  static constexpr mode_t kDirectoryMode = 040750;  // drwxr-x--- (directory)
  static constexpr mode_t kFileMode = 0100640;      // -rw-r----- (regular file)

  devtools_ijar::Stat file_stat;
  file_stat.total_size = info.Size();
  file_stat.is_directory = info.IsDirectory();
  file_stat.file_mode = info.IsDirectory() ? kDirectoryMode : kFileMode;
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
    bazel::windows::AutoHandle handle;
    Path path;
    if (!path.Set(root.Get() + L"\\" + files[i].RelativePath()) ||
        (!files[i].IsDirectory() && !OpenExistingFileForRead(path, &handle))) {
      LogError(__LINE__,
               (std::wstring(L"Failed to open file \"") + path.Get() + L"\"")
                   .c_str());
      return false;
    }

    devtools_ijar::u1* dest;
    if (!GetZipEntryPtr(zip_builder.get(), zip_entry_paths.EntryPathPtrs()[i],
                        GetZipAttr(files[i]), &dest) ||
        (!files[i].IsDirectory() &&
         !ReadCompleteFile(handle, dest, files[i].Size()))) {
      LogError(__LINE__, (std::wstring(L"Failed to dump file \"") + path.Get() +
                          L"\" into zip")
                             .c_str());
      return false;
    }

    if (zip_builder->FinishFile(files[i].Size(), /* compress */ false,
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
      LogError(__LINE__, (std::wstring(L"Failed to convert path \"") +
                          argv0.Get() + L"\"")
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
                     const Path& outerr, std::unique_ptr<Tee>* tee,
                     bazel::windows::AutoHandle* process) {
  SECURITY_ATTRIBUTES inheritable_handle_sa = {sizeof(SECURITY_ATTRIBUTES),
                                               NULL, TRUE};

  // Create a pipe to stream the output of the subprocess to this process.
  // The subprocess inherits two copies of the writing end (one for stdout, one
  // for stderr). This process closes its copies of the handles.
  // This process keeps the reading end and streams data from the pipe to the
  // test log and to stdout.
  HANDLE pipe_read_h, pipe_write_h;
  if (!CreatePipe(&pipe_read_h, &pipe_write_h, &inheritable_handle_sa, 0)) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "CreatePipe", err);
    return false;
  }
  bazel::windows::AutoHandle pipe_read(pipe_read_h), pipe_write(pipe_write_h);

  // Duplicate the write end of the pipe.
  // The original will be connected to the stdout of the process, the duplicate
  // to stderr.
  HANDLE pipe_write_dup_h;
  if (!DuplicateHandle(GetCurrentProcess(), pipe_write, GetCurrentProcess(),
                       &pipe_write_dup_h, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "DuplicateHandle", err);
    return false;
  }
  bazel::windows::AutoHandle pipe_write_dup(pipe_write_dup_h);

  // Open a readonly handle to NUL. The subprocess inherits this handle that's
  // connected to its stdin.
  bazel::windows::AutoHandle devnull_read(CreateFileW(
      L"NUL", GENERIC_READ,
      FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE,
      &inheritable_handle_sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
  if (devnull_read == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "CreateFileW", err);
    return false;
  }

  // Create an attribute object that specifies which particular handles shall
  // the subprocess inherit. We pass this object to CreateProcessW.
  HANDLE handle_array[] = {devnull_read, pipe_write, pipe_write_dup};
  std::unique_ptr<bazel::windows::AutoAttributeList> attr_list;
  std::wstring werror;
  if (!bazel::windows::AutoAttributeList::Create(handle_array, 3, &attr_list,
                                                 &werror)) {
    LogError(__LINE__, werror.c_str());
    return false;
  }

  // Open a handle to the test log file. The "tee" thread will write everything
  // into it that the subprocess writes to the pipe.
  bazel::windows::AutoHandle test_outerr;
  if (!OpenFileForWriting(outerr, &test_outerr)) {
    LogError(__LINE__, (std::wstring(L"Failed to open for writing \"") +
                        outerr.Get() + L"\"")
                           .c_str());
    return false;
  }

  // Duplicate stdout's handle, and pass it to the tee thread, who will own it
  // and close it in the end.
  HANDLE stdout_dup_h;
  if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE),
                       GetCurrentProcess(), &stdout_dup_h, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, "DuplicateHandle", err);
    return false;
  }
  bazel::windows::AutoHandle stdout_dup(stdout_dup_h);

  // Create the tee thread, and transfer ownerships of the `pipe_read`,
  // `test_outerr`, and `stdout_dup` handles.
  if (!TeeImpl::Create(&pipe_read, &test_outerr, &stdout_dup, tee)) {
    LogError(__LINE__);
    return false;
  }

  PROCESS_INFORMATION process_info;
  STARTUPINFOEXW startup_info;
  ZeroMemory(&startup_info, sizeof(STARTUPINFOW));
  startup_info.StartupInfo.cb = sizeof(STARTUPINFOEXW);
  startup_info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  // Do not Release() `devnull_read`, `pipe_write`, and `pipe_write_dup`. The
  // subprocess inherits a copy of these handles and we need to close them in
  // this process (via ~AutoHandle()).
  startup_info.StartupInfo.hStdInput = devnull_read;
  startup_info.StartupInfo.hStdOutput = pipe_write;
  startup_info.StartupInfo.hStdError = pipe_write_dup;
  startup_info.lpAttributeList = *attr_list.get();

  std::unique_ptr<WCHAR[]> cmdline;
  if (!CreateCommandLine(path, args, &cmdline)) {
    return false;
  }

  if (CreateProcessW(NULL, cmdline.get(), NULL, NULL, TRUE,
                     CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                     NULL, NULL, reinterpret_cast<STARTUPINFOW*>(&startup_info),
                     &process_info) != 0) {
    CloseHandle(process_info.hThread);
    *process = process_info.hProcess;
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

bool ArchiveUndeclaredOutputs(const UndeclaredOutputs& undecl) {
  if (undecl.root.Get().empty()) {
    // TEST_UNDECLARED_OUTPUTS_DIR was undefined, there's nothing to archive.
    return true;
  }

  std::vector<FileInfo> files;
  return GetFileListRelativeTo(undecl.root, &files) &&
         (files.empty() ||
          (CreateZip(undecl.root, files, undecl.zip) &&
           CreateUndeclaredOutputsManifest(files, undecl.manifest)));
}

// Creates the Undeclared Outputs Annotations file.
//
// This file is a concatenation of every *.part file directly under
// `undecl_annot_dir`. The file is written to `output`.
bool CreateUndeclaredOutputsAnnotations(const Path& undecl_annot_dir,
                                        const Path& output) {
  if (undecl_annot_dir.Get().empty()) {
    // The directory's environment variable
    // (TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR) was probably undefined, nothing
    // to do.
    return true;
  }

  std::vector<FileInfo> files;
  if (!GetFileListRelativeTo(undecl_annot_dir, &files, 0)) {
    LogError(__LINE__, (std::wstring(L"Failed to get files under \"") +
                        undecl_annot_dir.Get() + L"\"")
                           .c_str());
    return false;
  }
  // There are no *.part files under `undecl_annot_dir`, nothing to do.
  if (files.empty()) {
    return true;
  }

  bazel::windows::AutoHandle handle;
  if (!OpenFileForWriting(output, &handle)) {
    LogError(__LINE__, (std::wstring(L"Failed to open for writing \"") +
                        output.Get() + L"\"")
                           .c_str());
    return false;
  }

  for (const auto& e : files) {
    if (!e.IsDirectory() &&
        e.RelativePath().rfind(L".part") == e.RelativePath().size() - 5) {
      // Only consume "*.part" files.
      Path path;
      if (!path.Set(undecl_annot_dir.Get() + L"\\" + e.RelativePath()) ||
          !AppendFileTo(path, e.Size(), handle)) {
        LogError(__LINE__, (std::wstring(L"Failed to append file \"") +
                            path.Get() + L"\" to \"" + output.Get() + L"\"")
                               .c_str());
        return false;
      }
    }
  }
  return true;
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

bool TeeImpl::Create(bazel::windows::AutoHandle* input,
                     bazel::windows::AutoHandle* output1,
                     bazel::windows::AutoHandle* output2,
                     std::unique_ptr<Tee>* result) {
  std::unique_ptr<TeeImpl> tee(new TeeImpl(input, output1, output2));
  bazel::windows::AutoHandle thread(
      CreateThread(NULL, 0, ThreadFunc, tee.get(), 0, NULL));
  if (!thread.IsValid()) {
    return false;
  }
  result->reset(tee.release());
  return true;
}

DWORD WINAPI TeeImpl::ThreadFunc(LPVOID lpParam) {
  return reinterpret_cast<TeeImpl*>(lpParam)->MainFunc() ? 0 : 1;
}

bool TeeImpl::MainFunc() const {
  static constexpr size_t kBufferSize = 0x10000;
  DWORD read;
  uint8_t content[kBufferSize];
  while (ReadFile(input_, content, kBufferSize, &read, NULL)) {
    DWORD written;
    if (read > 0 && (!WriteFile(output1_, content, read, &written, NULL) ||
                     !WriteFile(output2_, content, read, &written, NULL))) {
      return false;
    }
  }
  return true;
}

int RunSubprocess(const Path& test_path,
                  const std::vector<const wchar_t*>& args,
                  const Path& test_outerr) {
  std::unique_ptr<Tee> tee;
  bazel::windows::AutoHandle process;
  if (!StartSubprocess(test_path, args, test_outerr, &tee, &process)) {
    return 1;
  }

  return WaitForSubprocess(process);
}

// Replace invalid XML characters and locate invalid CDATA sequences.
//
// The legal Unicode code points and ranges are U+0009, U+000A, U+000D,
// U+0020..U+D7FF, U+E000..U+FFFD, and U+10000..U+10FFFF.
//
// Assuming the input is UTF-8 encoded, that translates to the following
// regexps:
//   [\x9\xa\xd\x20-\x7f]                         <--- (9,A,D,20-7F)
//   [\xc0-\xdf][\x80-\xbf]                       <--- (0080-07FF)
//   [\xe0-\xec][\x80-\xbf][\x80-\xbf]            <--- (0800-CFFF)
//   [\xed][\x80-\x9f][\x80-\xbf]                 <--- (D000-D7FF)
//   [\xee][\x80-\xbf][\x80-\xbf]                 <--- (E000-EFFF)
//   [\xef][\x80-\xbe][\x80-\xbf]                 <--- (F000-FFEF)
//   [\xef][\xbf][\x80-\xbd]                      <--- (FFF0-FFFD)
//   [\xf0-\xf7][\x80-\xbf][\x80-\xbf][\x80-\xbf] <--- (010000-10FFFF)
//
// (See https://github.com/bazelbuild/bazel/issues/4691#issuecomment-408089257)
//
// Every octet-sequence matching one of these regexps will be left alone, all
// other octet-sequences will be replaced by '?' characters.
//
// This function also memorizes the locations of "]]>" in `cdata_end_locations`.
// The reason is "]]>" ends the CDATA section prematurely and cannot be escaped
// (see https://stackoverflow.com/a/223782/7778502). A separate filtering step
// can replace those sequences with the string "]]>]]&gt;<![CDATA[" (which ends
// the current CDATA segment, adds "]]&gt;", then starts a new CDATA segment).
void CdataEscape(uint8_t* p, const size_t size,
                 std::vector<uint8_t*>* cdata_end_locations) {
  for (size_t i = 0; i < size; ++i, ++p) {
    if (p[0] == ']' && (i + 2 < size) && p[1] == ']' && p[2] == '>') {
      // Mark where "]]>" is, then skip the next two octets.
      cdata_end_locations->push_back(p);
      i += 2;
      p += 2;
    } else if (*p == 0x9 || *p == 0xA || *p == 0xD ||
               (*p >= 0x20 && *p <= 0x7F)) {
      // Matched legal single-octet sequence. Nothing to do.
    } else if ((i + 1 < size) && p[0] >= 0xC0 && p[0] <= 0xDF && p[1] >= 0x80 &&
               p[1] <= 0xBF) {
      // Matched legal double-octet sequence. Skip the next octet.
      i += 1;
      p += 1;
    } else if ((i + 2 < size) &&
               ((p[0] >= 0xE0 && p[0] <= 0xEC && p[1] >= 0x80 && p[1] <= 0xBF &&
                 p[2] >= 0x80 && p[2] <= 0xBF) ||
                (p[0] == 0xED && p[1] >= 0x80 && p[1] <= 0x9F && p[2] >= 0x80 &&
                 p[2] <= 0xBF) ||
                (p[0] == 0xEE && p[1] >= 0x80 && p[1] <= 0xBF && p[2] >= 0x80 &&
                 p[2] <= 0xBF) ||
                (p[0] == 0xEF && p[1] >= 0x80 && p[1] <= 0xBE && p[2] >= 0x80 &&
                 p[2] <= 0xBF) ||
                (p[0] == 0xEF && p[1] == 0xBF && p[2] >= 0x80 &&
                 p[2] <= 0xBD))) {
      // Matched legal triple-octet sequence. Skip the next two octets.
      i += 2;
      p += 2;
    } else if ((i + 3 < size) && p[0] >= 0xF0 && p[0] <= 0xF7 && p[1] >= 0x80 &&
               p[1] <= 0xBF && p[2] >= 0x80 && p[2] <= 0xBF && p[3] >= 0x80 &&
               p[3] <= 0xBF) {
      // Matched legal quadruple-octet sequence. Skip the next three octets.
      i += 3;
      p += 3;
    } else {
      // Illegal octet; replace.
      *p = '?';
    }
  }
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
  Path test_path, exec_root, srcdir, tmpdir, test_outerr;
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
      !ExportXmlPath(exec_root, &test_outerr) ||
      !GetAndUnexportUndeclaredOutputsEnvvars(exec_root, &undecl)) {
    return 1;
  }

  int result = RunSubprocess(test_path, args, test_outerr);
  if (result != 0) {
    return result;
  }
  return (ArchiveUndeclaredOutputs(undecl) &&
          CreateUndeclaredOutputsAnnotations(undecl.annotations_dir,
                                             undecl.annotations))
             ? 0
             : 1;
}

namespace testing {

bool TestOnly_GetEnv(const wchar_t* name, std::wstring* result) {
  return GetEnv(name, result);
}

bool TestOnly_GetFileListRelativeTo(const std::wstring& abs_root,
                                    std::vector<FileInfo>* result,
                                    int depth_limit) {
  Path root;
  return blaze_util::IsAbsolute(abs_root) && root.Set(abs_root) &&
         GetFileListRelativeTo(root, result, depth_limit);
}

bool TestOnly_ToZipEntryPaths(const std::wstring& abs_root,
                              const std::vector<FileInfo>& files,
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

std::string TestOnly_GetMimeType(const std::string& filename) {
  return GetMimeType(filename);
}

bool TestOnly_CreateUndeclaredOutputsManifest(
    const std::vector<FileInfo>& files, std::string* result) {
  return CreateUndeclaredOutputsManifestContent(files, result);
}

bool TestOnly_CreateUndeclaredOutputsAnnotations(
    const std::wstring& abs_root, const std::wstring& abs_output) {
  Path root, output;
  return blaze_util::IsAbsolute(abs_root) && root.Set(abs_root) &&
         blaze_util::IsAbsolute(abs_output) && output.Set(abs_output) &&
         CreateUndeclaredOutputsAnnotations(root, output);
}

bool TestOnly_AsMixedPath(const std::wstring& path, std::string* result) {
  Path p;
  return p.Set(path) && WcsToAcp(AsMixedPath(RemoveUncPrefixMaybe(p)), result);
}

bool TestOnly_CreateTee(bazel::windows::AutoHandle* input,
                        bazel::windows::AutoHandle* output1,
                        bazel::windows::AutoHandle* output2,
                        std::unique_ptr<Tee>* result) {
  return TeeImpl::Create(input, output1, output2, result);
}

bool TestOnly_CdataEncodeBuffer(uint8_t* buffer, const size_t size,
                                std::vector<uint8_t*>* cdata_end_locations) {
  CdataEscape(buffer, size, cdata_end_locations);
  return true;
}

}  // namespace testing
}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
