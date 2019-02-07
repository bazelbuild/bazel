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
#include <string.h>
#include <sys/types.h>
#include <wchar.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
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

// Buffered input stream (based on a Windows HANDLE) with peek-ahead support.
//
// This class uses two consecutive "pages" where it buffers data from the
// underlying HANDLE (wrapped in an AutoHandle). Both pages are always loaded
// with data until there's no more data to read.
//
// The "active" page is the one where the read cursor is pointing. The other
// page is the next one to be read once the client moves the read cursor beyond
// the end of the active page.
//
// The client advances the read cursor with Advance(). When the cursor reaches
// the end of the active page, the other page becomes the active one (whose data
// is already buffered), and the old active page is loaded with new data from
// the underlying file.
class IFStreamImpl : IFStream {
 public:
  // Creates a new IFStream.
  //
  // If successful, then takes ownership of the HANDLE in 'handle', and returns
  // a new IFStream pointer. Otherwise leaves 'handle' alone and returns
  // nullptr.
  static IFStream* Create(bazel::windows::AutoHandle* handle,
                          DWORD max_page_size = 0x100000 /* 1 MB */);

  bool Get(uint8_t* result) const override;
  bool Advance() override;

 protected:
  bool PeekN(DWORD n, uint8_t* result) const override;

 private:
  bazel::windows::AutoHandle handle_;
  const std::unique_ptr<uint8_t[]> data_;
  const DWORD max_page_size_;
  DWORD page1_size_;
  DWORD page2_size_;
  DWORD page_end_;
  DWORD read_pos_;

  IFStreamImpl(bazel::windows::AutoHandle* handle,
               std::unique_ptr<uint8_t[]>&& data, DWORD data_size,
               DWORD max_page_size)
      : handle_(handle),
        data_(std::move(data)),
        max_page_size_(max_page_size),
        page1_size_(data_size > max_page_size ? max_page_size : data_size),
        page2_size_(data_size > max_page_size ? data_size - max_page_size : 0),
        read_pos_(0),
        page_end_(page1_size_) {}

  bool Page1Active() const { return read_pos_ < max_page_size_; }
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

struct Duration {
  static constexpr int kMax = INT_MAX;

  int seconds;

  bool FromString(const wchar_t* str);
};

void WriteStdout(const std::string& s) {
  DWORD written;
  WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), s.c_str(), s.size(), &written,
            NULL);
}

void LogError(const int line) {
  std::stringstream ss;
  ss << "ERROR(" << __FILE__ << ":" << line << ")" << std::endl;
  WriteStdout(ss.str());
}

void LogError(const int line, const std::string& msg) {
  std::stringstream ss;
  ss << "ERROR(" << __FILE__ << ":" << line << ") " << msg << std::endl;
  WriteStdout(ss.str());
}

void LogError(const int line, const std::wstring& msg) {
  std::string acp_msg;
  if (blaze_util::WcsToAcp(msg, &acp_msg)) {
    LogError(line, acp_msg);
  }
}

void LogErrorWithValue(const int line, const std::string& msg, DWORD value) {
  std::stringstream ss;
  ss << "value: " << value << " (0x";
  ss.setf(std::ios_base::hex, std::ios_base::basefield);
  ss << std::setw(8) << std::setfill('0') << value << "): ";
  ss.setf(std::ios_base::dec, std::ios_base::basefield);
  ss << msg;
  LogError(line, ss.str());
}

void LogErrorWithValue(const int line, const std::wstring& msg, DWORD value) {
  std::string acp_msg;
  if (blaze_util::WcsToAcp(msg, &acp_msg)) {
    LogErrorWithValue(line, acp_msg, value);
  }
}

void LogErrorWithArgAndValue(const int line, const std::string& msg,
                             const std::string& arg, DWORD value) {
  std::stringstream ss;
  ss << "value: " << value << " (0x";
  ss.setf(std::ios_base::hex, std::ios_base::basefield);
  ss << std::setw(8) << std::setfill('0') << value << "): argument: ";
  ss.setf(std::ios_base::dec, std::ios_base::basefield);
  ss << arg << ": " << msg;
  LogError(line, ss.str());
}

void LogErrorWithArgAndValue(const int line, const std::string& msg,
                             const std::wstring& arg, DWORD value) {
  std::string acp_arg;
  if (blaze_util::WcsToAcp(arg, &acp_arg)) {
    LogErrorWithArgAndValue(line, msg, acp_arg, value);
  }
}

std::wstring AddUncPrefixMaybe(const Path& p) {
  return bazel::windows::AddUncPrefixMaybe(p.Get());
}

std::wstring RemoveUncPrefixMaybe(const Path& p) {
  return bazel::windows::RemoveUncPrefixMaybe(p.Get());
}

inline bool CreateDirectories(const Path& path) {
  blaze_util::MakeDirectoriesW(AddUncPrefixMaybe(path), 0777);
  return true;
}

inline bool ToInt(const wchar_t* s, int* result) {
  return std::swscanf(s, L"%d", result) == 1;
}

bool WcsToAcp(const std::wstring& wcs, std::string* acp) {
  uint32_t err;
  if (!blaze_util::WcsToAcp(wcs, acp, &err)) {
    LogErrorWithArgAndValue(__LINE__, "Failed to convert string", wcs, err);
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
                            full_subdir, err);
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
                                  rel_path, 0);
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
                              unc_root + L"\\" + subdir, err);
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
             std::wstring(L"Failed to convert path \"") + root.Get() + L"\"");
    return false;
  }

  // Convert all UTF-16 paths to ANSI paths.
  std::vector<std::string> acp_file_list;
  acp_file_list.reserve(files.size());
  for (const auto& e : files) {
    std::string acp_path;
    if (!WcsToAcp(AsMixedPath(e.RelativePath()), &acp_path)) {
      LogError(__LINE__, std::wstring(L"Failed to convert path \"") +
                             e.RelativePath() + L"\"");
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
             std::wstring(L"Failed to convert path \"") + zip.Get() + L"\"");
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
    LogErrorWithArgAndValue(__LINE__, "Failed to open file", path.Get(), err);
    return false;
  }
  *result = h;
  return true;
}

bool OpenExistingFileForRead(const Path& abs_path,
                             bazel::windows::AutoHandle* result) {
  HANDLE h = CreateFileW(AddUncPrefixMaybe(abs_path).c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    LogErrorWithArgAndValue(__LINE__, "Failed to open file", abs_path.Get(),
                            err);
    return false;
  }
  *result = h;
  return true;
}

bool CreateEmptyFile(const Path& path) {
  bazel::windows::AutoHandle handle;
  return OpenFileForWriting(path, &handle);
}

bool ReadFromFile(HANDLE handle, uint8_t* dest, DWORD max_read) {
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

bool ReadCompleteFile(const Path& path, std::unique_ptr<uint8_t[]>* data,
                      DWORD* size) {
  bazel::windows::AutoHandle handle;
  if (!OpenExistingFileForRead(path, &handle)) {
    LogError(__LINE__, path.Get());
    return false;
  }

  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(handle, &file_size)) {
    DWORD err = GetLastError();
    LogErrorWithValue(__LINE__, path.Get(), err);
    return false;
  }

  // `ReadCompleteFile` doesn't support files larger than 4GB because most files
  // that this function will be reading (test outerr logs) are typically smaller
  // than that. (A buffered file reader would allow supporting larger files, but
  // that seems like overkill here.)
  if (file_size.QuadPart > 0xFFFFFFFF) {
    LogError(__LINE__, path.Get());
    return false;
  }
  const DWORD file_size_dw = file_size.QuadPart;
  *size = file_size_dw;

  // Allocate a buffer large enough to hold the whole file.
  data->reset(new uint8_t[file_size_dw]);
  if (!data->get()) {
    // Memory allocation failed.
    LogErrorWithValue(__LINE__, path.Get(), file_size_dw);
    return false;
  }

  return ReadFromFile(handle, data->get(), file_size_dw);
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
    LogError(__LINE__,
             std::wstring(L"Failed to open file \"") + file.Get() + L"\"");
    return false;
  }

  const size_t buf_size = std::min<size_t>(total_size, /* 10 MB */ 10000000);
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[buf_size]);

  while (true) {
    // Read at most `buf_size` many bytes from the input file.
    DWORD read = 0;
    if (!ReadFile(input, buffer.get(), buf_size, &read, NULL)) {
      DWORD err = GetLastError();
      LogErrorWithArgAndValue(__LINE__, "Failed to read file", file.Get(), err);
      return false;
    }
    if (read == 0) {
      // Reached end of input file.
      return true;
    }
    if (!WriteToFile(output, buffer.get(), read)) {
      LogError(__LINE__,
               std::wstring(L"Failed to append file \"") + file.Get() + L"\"");
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
             std::wstring(L"Failed to create manifest content for file \"") +
                 output.Get() + L"\"");
    return false;
  }

  bazel::windows::AutoHandle handle;
  if (!OpenFileForWriting(output, &handle)) {
    LogError(__LINE__, std::wstring(L"Failed to open file for writing \"") +
                           output.Get() + L"\"");
    return false;
  }

  if (!WriteToFile(handle, content.c_str(), content.size())) {
    LogError(__LINE__,
             std::wstring(L"Failed to write file \"") + output.Get() + L"\"");
    return false;
  }
  return true;
}

bool ExportXmlPath(const Path& cwd, Path* test_outerr, Path* xml_log) {
  if (!GetPathEnv(L"XML_OUTPUT_FILE", xml_log)) {
    LogError(__LINE__);
    return false;
  }
  xml_log->Absolutize(cwd);
  if (!test_outerr->Set(xml_log->Get() + L".log")) {
    LogError(__LINE__);
    return false;
  }
  std::wstring unix_result = AsMixedPath(xml_log->Get());
  return SetEnv(L"XML_OUTPUT_FILE", unix_result) &&
         // TODO(ulfjack): Update Gunit to accept XML_OUTPUT_FILE and drop the
         // GUNIT_OUTPUT env variable.
         SetEnv(L"GUNIT_OUTPUT", L"xml:" + unix_result) &&
         CreateDirectories(xml_log->Dirname()) && CreateEmptyFile(*test_outerr);
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
    LogError(__LINE__, std::string("Failed to add new zip entry for file \"") +
                           entry_name + "\": " + zip_builder->GetError());
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
               std::wstring(L"Failed to open file \"") + path.Get() + L"\"");
      return false;
    }

    devtools_ijar::u1* dest;
    if (!GetZipEntryPtr(zip_builder.get(), zip_entry_paths.EntryPathPtrs()[i],
                        GetZipAttr(files[i]), &dest) ||
        (!files[i].IsDirectory() &&
         !ReadFromFile(handle, dest, files[i].Size()))) {
      LogError(__LINE__, std::wstring(L"Failed to dump file \"") + path.Get() +
                             L"\" into zip");
      return false;
    }

    if (zip_builder->FinishFile(files[i].Size(), /* compress */ false,
                                /* compute_crc */ true) == -1) {
      LogError(__LINE__, std::wstring(L"Failed to finish writing file \"") +
                             path.Get() + L"\" to zip");
      return false;
    }
  }

  if (zip_builder->Finish() == -1) {
    LogError(__LINE__, std::string("Failed to add file to zip: ") +
                           zip_builder->GetError());
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

bool PrintTestLogStartMarker() {
  std::wstring test_target;
  std::string acp_test_target;
  if (!GetEnv(L"TEST_TARGET", &test_target) ||
      !WcsToAcp(test_target, &acp_test_target)) {
    return false;
  }
  std::stringstream ss;
  if (test_target.empty()) {
    // According to the Bazel Test Encyclopedia, setting TEST_TARGET is
    // optional.
    ss << "Executing tests from unknown target\n";
  } else {
    ss << "Executing tests from " << acp_test_target << "\n";
  }

  // This header marks where --test_output=streamed will start being printed.
  ss << "---------------------------------------------------------------------"
        "--------\n";
  WriteStdout(ss.str());
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
      LogError(__LINE__, std::wstring(L"Failed to convert path \"") +
                             argv0.Get() + L"\"");
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
                              test_path, err);
      return false;
    }

    std::string rloc = runfiles->Rlocation(utf8_test_path);
    if (!blaze_util::Utf8ToWcs(rloc, &test_path, &err)) {
      LogErrorWithArgAndValue(__LINE__, "Failed to convert string",
                              utf8_test_path, err);
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
               std::wstring(L"Failed to add command line argument \"") + arg +
                   L"\"; command would be too long");
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
               std::wstring(L"Failed to add command line argument \"") + arg +
                   L"\"; command would be too long");
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
                     LARGE_INTEGER* start_time,
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
  std::unique_ptr<bazel::windows::AutoAttributeList> attr_list;
  std::wstring werror;
  if (!bazel::windows::AutoAttributeList::Create(
          devnull_read, pipe_write, pipe_write_dup, &attr_list, &werror)) {
    LogError(__LINE__, werror);
    return false;
  }

  // Open a handle to the test log file. The "tee" thread will write everything
  // into it that the subprocess writes to the pipe.
  bazel::windows::AutoHandle test_outerr;
  if (!OpenFileForWriting(outerr, &test_outerr)) {
    LogError(__LINE__, std::wstring(L"Failed to open for writing \"") +
                           outerr.Get() + L"\"");
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
  attr_list->InitStartupInfoExW(&startup_info);

  std::unique_ptr<WCHAR[]> cmdline;
  if (!CreateCommandLine(path, args, &cmdline)) {
    return false;
  }

  QueryPerformanceCounter(start_time);
  if (CreateProcessW(NULL, cmdline.get(), NULL, NULL, TRUE,
                     CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                     NULL, NULL, &startup_info.StartupInfo,
                     &process_info) != 0) {
    CloseHandle(process_info.hThread);
    *process = process_info.hProcess;
    return true;
  } else {
    DWORD err = GetLastError();
    LogErrorWithValue(
        __LINE__,
        (std::wstring(L"CreateProcessW failed (") + cmdline.get() + L")")
            .c_str(),
        err);
    return false;
  }
}

int WaitForSubprocess(HANDLE process, LARGE_INTEGER* end_time) {
  DWORD result = WaitForSingleObject(process, INFINITE);
  QueryPerformanceCounter(end_time);
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
    LogError(__LINE__, std::wstring(L"Failed to get files under \"") +
                           undecl_annot_dir.Get() + L"\"");
    return false;
  }
  // There are no *.part files under `undecl_annot_dir`, nothing to do.
  if (files.empty()) {
    return true;
  }

  bazel::windows::AutoHandle handle;
  if (!OpenFileForWriting(output, &handle)) {
    LogError(__LINE__, std::wstring(L"Failed to open for writing \"") +
                           output.Get() + L"\"");
    return false;
  }

  for (const auto& e : files) {
    if (!e.IsDirectory() &&
        e.RelativePath().rfind(L".part") == e.RelativePath().size() - 5) {
      // Only consume "*.part" files.
      Path path;
      if (!path.Set(undecl_annot_dir.Get() + L"\\" + e.RelativePath()) ||
          !AppendFileTo(path, e.Size(), handle)) {
        LogError(__LINE__, std::wstring(L"Failed to append file \"") +
                               path.Get() + L"\" to \"" + output.Get() + L"\"");
        return false;
      }
    }
  }
  return true;
}

bool ParseArgs(int argc, wchar_t** argv, Path* out_argv0,
               std::wstring* out_test_path_arg,
               std::vector<const wchar_t*>* out_args) {
  if (!out_argv0->Set(argv[0])) {
    return false;
  }
  argc--;
  argv++;

  if (argc < 1) {
    LogError(__LINE__, "Usage: $0 <test_path> [test_args...]");
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

bool ParseXmlWriterArgs(int argc, wchar_t** argv, const Path& cwd,
                        Path* out_test_log, Path* out_xml_log,
                        Duration* out_duration, int* out_exit_code) {
  if (argc < 5) {
    LogError(__LINE__,
             "Usage: $0 <test_output_path> <xml_log_path>"
             " <duration_in_seconds> <exit_code>");
    return false;
  }
  if (!out_test_log->Set(argv[1]) || out_test_log->Get().empty()) {
    LogError(__LINE__, (std::wstring(L"Failed to parse test log path from \"") +
                        argv[1] + L"\"")
                           .c_str());
    return false;
  }
  out_test_log->Absolutize(cwd);
  if (!out_xml_log->Set(argv[2]) || out_xml_log->Get().empty()) {
    LogError(__LINE__, (std::wstring(L"Failed to parse XML log path from \"") +
                        argv[2] + L"\"")
                           .c_str());
    return false;
  }
  out_xml_log->Absolutize(cwd);
  if (!out_duration->FromString(argv[3])) {
    LogError(__LINE__, (std::wstring(L"Failed to parse test duration from \"") +
                        argv[3] + L"\"")
                           .c_str());
    return false;
  }
  if (!ToInt(argv[4], out_exit_code)) {
    LogError(__LINE__, (std::wstring(L"Failed to parse exit code from \"") +
                        argv[4] + L"\"")
                           .c_str());
    return false;
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
                  const Path& test_outerr, Duration* test_duration) {
  std::unique_ptr<Tee> tee;
  bazel::windows::AutoHandle process;
  LARGE_INTEGER start, end, freq;
  if (!StartSubprocess(test_path, args, test_outerr, &tee, &start, &process)) {
    LogError(__LINE__, std::wstring(L"Failed to start test process \"") +
                           test_path.Get() + L"\"");
    return 1;
  }
  int result = WaitForSubprocess(process, &end);

  QueryPerformanceFrequency(&freq);
  end.QuadPart -= start.QuadPart;
  decltype(LARGE_INTEGER::QuadPart) seconds;
  // Compute the number of seconds the test ran for.
  seconds = end.QuadPart / freq.QuadPart;
  // Check the remainder: if it's at least 0.5 seconds, round up.
  if ((end.QuadPart - seconds * freq.QuadPart) * 2 >= freq.QuadPart) {
    seconds += 1;
  }
  test_duration->seconds =
      (seconds > Duration::kMax) ? Duration::kMax : seconds;
  return result;
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
bool CdataEscape(const uint8_t* input, const DWORD size,
                 std::basic_ostream<char>* out) {
  // We aren't modifying the input, so const_cast is fine.
  uint8_t* p = const_cast<uint8_t*>(input);

  for (DWORD i = 0; i < size; ++i, ++p) {
    if (p[0] == ']' && (i + 2 < size) && p[1] == ']' && p[2] == '>') {
      *out << "]]>]]<![CDATA[>";
      if (!out->good()) {
        return false;
      }
      i += 2;
      p += 2;
    } else if (*p == 0x9 || *p == 0xA || *p == 0xD ||
               (*p >= 0x20 && *p <= 0x7F)) {
      // Matched legal single-octet sequence.
      *out << *p;
      if (!out->good()) {
        return false;
      }
    } else if ((i + 1 < size) && p[0] >= 0xC0 && p[0] <= 0xDF && p[1] >= 0x80 &&
               p[1] <= 0xBF) {
      // Matched legal double-octet sequence. Skip the next octet.
      *out << p[0] << p[1];
      if (!out->good()) {
        return false;
      }
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
      *out << p[0] << p[1] << p[2];
      if (!out->good()) {
        return false;
      }
      i += 2;
      p += 2;
    } else if ((i + 3 < size) && p[0] >= 0xF0 && p[0] <= 0xF7 && p[1] >= 0x80 &&
               p[1] <= 0xBF && p[2] >= 0x80 && p[2] <= 0xBF && p[3] >= 0x80 &&
               p[3] <= 0xBF) {
      // Matched legal quadruple-octet sequence. Skip the next three octets.
      *out << p[0] << p[1] << p[2] << p[3];
      if (!out->good()) {
        return false;
      }
      i += 3;
      p += 3;
    } else {
      // Illegal octet; replace.
      *out << '?';
      if (!out->good()) {
        return false;
      }
    }
  }
  return true;
}

bool CdataEscapeAndAppend(const Path& input, std::ofstream* out_stm) {
  DWORD size;
  std::unique_ptr<uint8_t[]> data;
  if (!ReadCompleteFile(input, &data, &size)) {
    LogError(__LINE__, input.Get());
    return false;
  }

  return CdataEscape(data.get(), size, out_stm);
}

bool GetTestName(std::wstring* result) {
  if (!GetEnv(L"TEST_BINARY", result) || result->empty()) {
    LogError(__LINE__, L"Failed to get test name");
    return false;
  }
  if (result->size() >= 2 && (*result)[0] == '.' && (*result)[1] == '/') {
    result->erase(0, 2);
  }

  // Ensure that test shards have unique names in the xml output, by including
  // the shard index in the test name.
  std::wstring total_shards_str;
  int total_shards = 0, shard_index = 0;
  if (!GetIntEnv(L"TEST_TOTAL_SHARDS", &total_shards_str, &total_shards)) {
    LogError(__LINE__);
    return false;
  }
  if (total_shards > 0) {
    std::wstring shard_index_str;
    if (!GetIntEnv(L"TEST_SHARD_INDEX", &shard_index_str, &shard_index) ||
        shard_index_str.empty()) {
      LogError(__LINE__);
      return false;
    }
    std::wstringstream stm;
    stm << *result << L"_shard_" << (shard_index + 1) << L"/"
        << total_shards_str;
    *result = stm.str();
  }

  return true;
}

std::string CreateErrorTag(int exit_code) {
  if (exit_code != 0) {
    std::stringstream ss;
    ss << "<error message=\"exited with error code " << exit_code
       << "\"></error>";
    return ss.str();
  } else {
    return std::string();
  }
}

bool ShouldCreateXml(const Path& xml_log, bool* result) {
  *result = true;

  DWORD attr = GetFileAttributesW(AddUncPrefixMaybe(xml_log).c_str());
  if (attr != INVALID_FILE_ATTRIBUTES) {
    // The XML file already exists, maybe the test framework wrote it.
    // Leave the file alone.
    *result = false;
    return true;
  }

  std::wstring split_xml_generation;
  if (!GetEnv(L"EXPERIMENTAL_SPLIT_XML_GENERATION", &split_xml_generation)) {
    LogError(__LINE__, "Failed to get %EXPERIMENTAL_SPLIT_XML_GENERATION%");
    return false;
  }
  if (split_xml_generation == L"1") {
    // Bazel generates the test xml as a separate action, so we don't have to
    // create it.
    *result = false;
  }

  return true;
}

bool CreateXmlLog(const Path& output, const Path& test_outerr,
                  const Duration duration, const int exit_code,
                  const bool delete_afterwards) {
  bool should_create_xml;
  if (!ShouldCreateXml(output, &should_create_xml)) {
    LogError(__LINE__,
             (std::wstring(L"CreateXmlLog(") + output.Get() + L")").c_str());
    return false;
  }
  if (!should_create_xml) {
    return true;
  }

  Defer delete_test_outerr([test_outerr, delete_afterwards]() {
    // Delete the test's outerr file after we have the XML file.
    // We don't care if this succeeds or not, because the outerr file is not a
    // declared output.
    if (delete_afterwards) {
      DeleteFileW(test_outerr.Get().c_str());
    }
  });

  std::wstring test_name;
  int errors = (exit_code == 0) ? 0 : 1;
  std::string error_msg = CreateErrorTag(exit_code);
  if (!GetTestName(&test_name)) {
    LogError(__LINE__);
    return false;
  }

  std::string acp_test_name;
  if (!WcsToAcp(test_name, &acp_test_name)) {
    LogError(__LINE__, test_name.c_str());
    return false;
  }

  std::ofstream stm(AddUncPrefixMaybe(output).c_str(),
                    std::ios_base::out | std::ios_base::binary
                        | std::ios_base::trunc);
  if (!stm.is_open() || !stm.good()) {
    LogError(__LINE__, output.Get().c_str());
    return false;
  }

  // Create XML file stub.
  stm << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         "<testsuites>\n"
         "<testsuite name=\""
      << acp_test_name << "\" tests=\"1\" failures=\"0\" errors=\"" << errors
      << "\">\n"
         "<testcase name=\""
      << acp_test_name << "\" status=\"run\" duration=\"" << duration.seconds
      << "\" time=\"" << duration.seconds << "\">" << error_msg
      << "</testcase>\n"
         "<system-out><![CDATA[";
  if (!stm.good()) {
    LogError(__LINE__, output.Get().c_str());
    return false;
  }

  // Encode test log to make it embeddable in CDATA.
  if (!CdataEscapeAndAppend(test_outerr, &stm)) {
    LogError(__LINE__, output.Get().c_str());
    return false;
  }

  // Append CDATA end and closing tags.
  stm << "]]></system-out>\n</testsuite>\n</testsuites>\n";
  if (!stm.good()) {
    LogError(__LINE__, output.Get().c_str());
    return false;
  }
  return true;
}

bool Duration::FromString(const wchar_t* str) {
  int result;
  if (!ToInt(str, &result)) {
    LogError(
        __LINE__,
        (std::wstring(L"Failed to parse int from \"") + str + L"\"").c_str());
    return false;
  }
  this->seconds = result;
  return true;
}

bool Path::Set(const std::wstring& path) {
  std::wstring result;
  std::string error;
  if (!blaze_util::AsWindowsPath(path, &result, &error)) {
    LogError(__LINE__, error);
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

IFStream* IFStreamImpl::Create(bazel::windows::AutoHandle* handle,
                               DWORD max_page_size) {
  std::unique_ptr<uint8_t[]> data(new uint8_t[max_page_size * 2]);
  DWORD read;
  if (!ReadFile(*handle, data.get(), max_page_size * 2, &read, NULL)) {
    DWORD err = GetLastError();
    if (err == ERROR_BROKEN_PIPE) {
      read = 0;
    } else {
      LogErrorWithValue(__LINE__, "Failed to read from file", err);
      return nullptr;
    }
  }
  return new IFStreamImpl(handle, std::move(data), read, max_page_size);
}

bool IFStreamImpl::Get(uint8_t* result) const {
  if (read_pos_ < page_end_) {
    *result = data_[read_pos_];
    return true;
  } else {
    return false;
  }
}

bool IFStreamImpl::Advance() {
  if (read_pos_ + 1 < page_end_) {
    read_pos_++;
    return true;
  }
  const bool page1_was_active = Page1Active();
  // The new page should have already been loaded when we started reading the
  // current one (or it was filled by the Create method). Its size should only
  // be zero if we reached EOF.
  if ((page1_was_active && page2_size_ == 0) ||
      (!page1_was_active && page1_size_ == 0)) {
    return false;
  }
  // Overwrite the *active* page, because read_pos_ is about to move out of it
  // and the current inactive page will be the new active one.
  if (!ReadFile(handle_,
                page1_was_active ? data_.get() : (data_.get() + max_page_size_),
                max_page_size_, page1_was_active ? &page1_size_ : &page2_size_,
                NULL)) {
    DWORD err = GetLastError();
    if (err == ERROR_BROKEN_PIPE) {
      // The stream is reading from a pipe, and there's no more data.
      if (page1_was_active) {
        page1_size_ = 0;
      } else {
        page2_size_ = 0;
      }
    } else {
      LogErrorWithValue(__LINE__, "Failed to read from file", err);
      return false;
    }
  }
  page_end_ = page1_was_active ? max_page_size_ + page2_size_ : page1_size_;
  read_pos_ = page1_was_active ? max_page_size_ : 0;
  return true;
}

bool IFStreamImpl::PeekN(DWORD n, uint8_t* result) const {
  if (n > 3) {
    // We only need to support peeking at up to 3 bytes. The theoretical upper
    // limit is max_page_size_ * 2 - 1, because the buffer can hold at most
    // max_page_size_ * 2 bytes of data and peeking starts at the next byte.
    return false;
  }

  if (page_end_ - read_pos_ > n) {
    // The current page has enough data we can peek at.
    for (DWORD i = 0; i < n; ++i) {
      result[i] = data_[read_pos_ + 1 + i];
    }
    return true;
  }
  DWORD required_from_next_page = n - (page_end_ - 1 - read_pos_);
  // Check that the next page has enough data.
  if ((Page1Active() && page2_size_ < required_from_next_page) ||
      (!Page1Active() && page1_size_ < required_from_next_page)) {
    // Pages are loaded eagerly by Advance(). The only way the next page's size
    // can be zero is if we reached EOF.
    return false;
  }
  for (DWORD i = 0, pos = read_pos_ + 1; i < n; ++i, ++pos) {
    if (pos == page_end_) {
      pos = Page1Active() ? max_page_size_ : 0;
    }
    result[i] = data_[pos];
  }
  return true;
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

int TestWrapperMain(int argc, wchar_t** argv) {
  Path argv0;
  std::wstring test_path_arg;
  Path test_path, exec_root, srcdir, tmpdir, test_outerr, xml_log;
  UndeclaredOutputs undecl;
  std::vector<const wchar_t*> args;
  if (!ParseArgs(argc, argv, &argv0, &test_path_arg, &args) ||
      !PrintTestLogStartMarker() ||
      !FindTestBinary(argv0, test_path_arg, &test_path) ||
      !GetCwd(&exec_root) || !ExportUserName() ||
      !ExportSrcPath(exec_root, &srcdir) ||
      !ExportTmpPath(exec_root, &tmpdir) || !ExportHome(tmpdir) ||
      !ExportRunfiles(exec_root, srcdir) || !ExportShardStatusFile(exec_root) ||
      !ExportGtestVariables(tmpdir) || !ExportMiscEnvvars(exec_root) ||
      !ExportXmlPath(exec_root, &test_outerr, &xml_log) ||
      !GetAndUnexportUndeclaredOutputsEnvvars(exec_root, &undecl)) {
    return 1;
  }

  Duration test_duration;
  int result = RunSubprocess(test_path, args, test_outerr, &test_duration);
  if (!CreateXmlLog(xml_log, test_outerr, test_duration, result, true) ||
      !ArchiveUndeclaredOutputs(undecl) ||
      !CreateUndeclaredOutputsAnnotations(undecl.annotations_dir,
                                          undecl.annotations)) {
    return 1;
  }
  return result;
}

int XmlWriterMain(int argc, wchar_t** argv) {
  Path cwd, test_outerr, test_xml_log;
  Duration duration;
  int exit_code = 0;

  if (!GetCwd(&cwd) ||
      !ParseXmlWriterArgs(argc, argv, cwd, &test_outerr, &test_xml_log,
                          &duration, &exit_code) ||
      !CreateXmlLog(test_xml_log, test_outerr, duration, exit_code, false)) {
    return 1;
  }

  return 0;
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

bool TestOnly_CdataEncode(const uint8_t* input, const DWORD size,
                          std::basic_ostream<char>* out_stm) {
  return CdataEscape(input, size, out_stm);
}

IFStream* TestOnly_CreateIFStream(bazel::windows::AutoHandle* handle,
                                  DWORD page_size) {
  return IFStreamImpl::Create(handle, page_size);
}

}  // namespace testing
}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
