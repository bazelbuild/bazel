// Copyright 2016 The Bazel Authors. All rights reserved.
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
#include <ctype.h>   // isalpha
#include <wchar.h>   // wcslen
#include <wctype.h>  // iswalpha
#include <windows.h>

#include <memory>  // unique_ptr
#include <sstream>
#include <string>
#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"

namespace blaze_util {

using bazel::windows::AutoHandle;
using bazel::windows::GetLongPath;
using bazel::windows::HasUncPrefix;
using std::basic_string;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;
using std::wstring;

static constexpr DWORD kAllShare =
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

// Returns true if `path` refers to a directory or (non-dangling) junction.
// `path` must be a normalized Windows path, with UNC prefix (and absolute) if
// necessary.
bool IsDirectoryW(const wstring& path);

// Returns true the file or junction at `path` is successfully deleted.
// Returns false otherwise, or if `path` doesn't exist or is a directory.
// `path` must be a normalized Windows path, with UNC prefix (and absolute) if
// necessary.
static bool UnlinkPathW(const wstring& path);

static bool CanReadFileW(const wstring& path);

template <typename char_type>
static bool IsPathSeparator(char_type ch) {
  return ch == '/' || ch == '\\';
}

class WindowsPipe : public IPipe {
 public:
  WindowsPipe(const HANDLE& read_handle, const HANDLE& write_handle)
      : _read_handle(read_handle), _write_handle(write_handle) {}

  WindowsPipe() = delete;

  bool Send(const void* buffer, int size) override {
    DWORD actually_written = 0;
    return ::WriteFile(_write_handle, buffer, size, &actually_written,
                       NULL) == TRUE;
  }

  int Receive(void* buffer, int size, int* error) override {
    DWORD actually_read = 0;
    BOOL result = ::ReadFile(_read_handle, buffer, size, &actually_read, NULL);
    if (error != nullptr) {
      // TODO(laszlocsomor): handle the error mode that is errno=EINTR on Linux.
      *error = result ? IPipe::SUCCESS : IPipe::OTHER_ERROR;
    }
    return result ? actually_read : -1;
  }

 private:
  AutoHandle _read_handle;
  AutoHandle _write_handle;
};

IPipe* CreatePipe() {
  // The pipe HANDLEs can be inherited.
  SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
  HANDLE read_handle = INVALID_HANDLE_VALUE;
  HANDLE write_handle = INVALID_HANDLE_VALUE;
  if (!CreatePipe(&read_handle, &write_handle, &sa, 0)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "CreatePipe failed: " << GetLastErrorString();
  }
  return new WindowsPipe(read_handle, write_handle);
}

class WindowsFileMtime : public IFileMtime {
 public:
  WindowsFileMtime()
      : near_future_(GetFuture(9)), distant_future_(GetFuture(10)) {}

  bool IsUntampered(const Path& path) override;
  bool SetToNow(const Path& path) override;
  bool SetToDistantFuture(const Path& path) override;

 private:
  // 9 years in the future.
  const FILETIME near_future_;
  // 10 years in the future.
  const FILETIME distant_future_;

  static FILETIME GetNow();
  static FILETIME GetFuture(WORD years);
  static bool Set(const Path& path, FILETIME time);
};

bool WindowsFileMtime::IsUntampered(const Path& path) {
  if (path.IsEmpty() || path.IsNull()) {
    return false;
  }

  // Get attributes, to check if the file exists. (It may still be a dangling
  // junction.)
  DWORD attrs = GetFileAttributesW(path.AsNativePath().c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return false;
  }

  bool is_directory = attrs & FILE_ATTRIBUTE_DIRECTORY;
  AutoHandle handle(CreateFileW(
      /* lpFileName */ path.AsNativePath().c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */
      // Per CreateFile's documentation on MSDN, opening directories requires
      // the FILE_FLAG_BACKUP_SEMANTICS flag.
      is_directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));

  if (!handle.IsValid()) {
    return false;
  }

  if (is_directory) {
    return true;
  } else {
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(handle, &info)) {
      return false;
    }

    // Compare the mtime with `near_future_`, not with `GetNow()` or
    // `distant_future_`.
    // This way we don't need to call GetNow() every time we want to compare
    // (and thus convert a SYSTEMTIME to FILETIME), and we also don't need to
    // worry about potentially unreliable FILETIME equality check (in case it
    // uses floats or something crazy).
    return CompareFileTime(&near_future_, &info.ftLastWriteTime) == -1;
  }
}

bool WindowsFileMtime::SetToNow(const Path& path) {
  return Set(path, GetNow());
}

bool WindowsFileMtime::SetToDistantFuture(const Path& path) {
  return Set(path, distant_future_);
}

bool WindowsFileMtime::Set(const Path& path, FILETIME time) {
  AutoHandle handle(::CreateFileW(
      /* lpFileName */ path.AsNativePath().c_str(),
      /* dwDesiredAccess */ FILE_WRITE_ATTRIBUTES,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */
      IsDirectoryW(path.AsNativePath())
          ? (FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
          : FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));
  if (!handle.IsValid()) {
    return false;
  }
  return ::SetFileTime(
             /* hFile */ handle,
             /* lpCreationTime */ NULL,
             /* lpLastAccessTime */ NULL,
             /* lpLastWriteTime */ &time) == TRUE;
}

FILETIME WindowsFileMtime::GetNow() {
  FILETIME now;
  GetSystemTimeAsFileTime(&now);
  return now;
}

FILETIME WindowsFileMtime::GetFuture(WORD years) {
  FILETIME result;
  GetSystemTimeAsFileTime(&result);

  // 1 year in FILETIME.
  constexpr ULONGLONG kOneYear = 365ULL * 24 * 60 * 60 * 10'000'000;

  ULARGE_INTEGER result_value;
  result_value.LowPart = result.dwLowDateTime;
  result_value.HighPart = result.dwHighDateTime;
  result_value.QuadPart += kOneYear * years;
  result.dwLowDateTime = result_value.LowPart;
  result.dwHighDateTime = result_value.HighPart;
  return result;
}

IFileMtime* CreateFileMtime() { return new WindowsFileMtime(); }

static bool OpenFileForReading(const Path& path, HANDLE* result) {
  *result = ::CreateFileW(
      /* lpFileName */ path.AsNativePath().c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ kAllShare,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  return true;
}

int ReadFromHandle(file_handle_type handle, void* data, size_t size,
                   int* error) {
  DWORD actually_read = 0;
  bool success = ::ReadFile(handle, data, size, &actually_read, NULL);
  if (error != nullptr) {
    // TODO(laszlocsomor): handle the error cases that are errno=EINTR and
    // errno=EAGAIN on Linux.
    *error = success ? ReadFileResult::SUCCESS : ReadFileResult::OTHER_ERROR;
  }
  return success ? actually_read : -1;
}

bool ReadFile(const string& filename, string* content, int max_size) {
  if (IsDevNull(filename.c_str())) {
    // mimic read(2) behavior: we can always read 0 bytes from /dev/null
    content->clear();
    return true;
  }
  return ReadFile(Path(filename), content, max_size);
}

bool ReadFile(const Path& path, std::string* content, int max_size) {
  if (path.IsEmpty()) {
    return false;
  }
  // TODO(laszlocsomor): remove the following check; it won't allow opening NUL.
  if (path.IsNull()) {
    return true;
  }

  HANDLE handle;
  if (!OpenFileForReading(path, &handle)) {
    return false;
  }

  AutoHandle autohandle(handle);
  if (!autohandle.IsValid()) {
    return false;
  }
  content->clear();
  return ReadFrom(handle, content, max_size);
}

bool ReadFile(const string& filename, void* data, size_t size) {
  return ReadFile(Path(filename), data, size);
}

bool ReadFile(const Path& path, void* data, size_t size) {
  if (path.IsEmpty()) {
    return false;
  }
  if (path.IsNull()) {
    // mimic read(2) behavior: we can always read 0 bytes from /dev/null
    return true;
  }
  HANDLE handle;
  if (!OpenFileForReading(path, &handle)) {
    return false;
  }

  AutoHandle autohandle(handle);
  if (!autohandle.IsValid()) {
    return false;
  }
  return ReadFrom(handle, data, size);
}

bool WriteFile(const void* data, size_t size, const string& filename,
               unsigned int perm) {
  if (IsDevNull(filename.c_str())) {
    return true;  // mimic write(2) behavior with /dev/null
  }
  return WriteFile(data, size, Path(filename), perm);
}

bool WriteFile(const void* data, size_t size, const Path& path,
               unsigned int perm) {
  UnlinkPathW(path.AsNativePath());  // We don't care about the success of this.
  AutoHandle handle(::CreateFileW(
      /* lpFileName */ path.AsNativePath().c_str(),
      /* dwDesiredAccess */ GENERIC_WRITE,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ CREATE_ALWAYS,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));
  if (!handle.IsValid()) {
    return false;
  }

  // TODO(laszlocsomor): respect `perm` and set the file permissions accordingly
  DWORD actually_written = 0;
  ::WriteFile(handle, data, size, &actually_written, NULL);
  return actually_written == size;
}

int WriteToStdOutErr(const void* data, size_t size, bool to_stdout) {
  DWORD written = 0;
  HANDLE h = ::GetStdHandle(to_stdout ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
  if (h == INVALID_HANDLE_VALUE) {
    return WriteResult::OTHER_ERROR;
  }

  if (::WriteFile(h, data, size, &written, NULL)) {
    return (written == size) ? WriteResult::SUCCESS : WriteResult::OTHER_ERROR;
  } else {
    return (GetLastError() == ERROR_NO_DATA) ? WriteResult::BROKEN_PIPE
                                             : WriteResult::OTHER_ERROR;
  }
}

int RenameDirectory(const std::string& old_name, const std::string& new_name) {
  wstring wold_name;
  string error;
  if (!AsAbsoluteWindowsPath(old_name, &wold_name, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "RenameDirectory(" << old_name << ", " << new_name
        << "): AsAbsoluteWindowsPath(" << old_name << ") failed: " << error;
    return kRenameDirectoryFailureOtherError;
  }

  wstring wnew_name;
  if (!AsAbsoluteWindowsPath(new_name, &wnew_name, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "RenameDirectory(" << old_name << ", " << new_name
        << "): AsAbsoluteWindowsPath(" << new_name << ") failed: " << error;
    return kRenameDirectoryFailureOtherError;
  }

  if (!::MoveFileExW(wold_name.c_str(), wnew_name.c_str(),
                     MOVEFILE_COPY_ALLOWED | MOVEFILE_FAIL_IF_NOT_TRACKABLE |
                         MOVEFILE_WRITE_THROUGH)) {
    return GetLastError() == ERROR_ALREADY_EXISTS
               ? kRenameDirectoryFailureNotEmpty
               : kRenameDirectoryFailureOtherError;
  }
  return kRenameDirectorySuccess;
}

static bool UnlinkPathW(const wstring& path) {
  DWORD attrs = ::GetFileAttributesW(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    // Path does not exist.
    return false;
  }
  if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
    if (!(attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
      // Path is a directory; unlink(2) also cannot remove directories.
      return false;
    }
    // Otherwise it's a junction, remove using RemoveDirectoryW.
    return ::RemoveDirectoryW(path.c_str()) == TRUE;
  } else {
    // Otherwise it's a file, remove using DeleteFileW.
    return ::DeleteFileW(path.c_str()) == TRUE;
  }
}

bool UnlinkPath(const string& file_path) {
  if (IsDevNull(file_path.c_str())) {
    return false;
  }
  return UnlinkPath(Path(file_path));
}

bool UnlinkPath(const Path& path) { return UnlinkPathW(path.AsNativePath()); }

static bool RealPath(const WCHAR* path, unique_ptr<WCHAR[]>* result = nullptr) {
  // Attempt opening the path, which may be anything -- a file, a directory, a
  // symlink, even a dangling symlink is fine.
  // Follow reparse points, getting us that much closer to the real path.
  AutoHandle h(CreateFileW(path, 0, kAllShare, NULL, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, NULL));
  if (!h.IsValid()) {
    // Path does not exist or it's a dangling junction/symlink.
    return false;
  }

  if (!result) {
    // The caller is only interested in whether the file exists, they aren't
    // interested in its real path. Since we just successfully opened the file
    // we already know it exists.
    // Also, GetFinalPathNameByHandleW is slow so avoid calling it if we can.
    return true;
  }

  // kMaxPath value: according to MSDN, maximum path length is 32767, and with
  // an extra null terminator that's exactly 0x8000.
  static constexpr size_t kMaxPath = 0x8000;
  std::unique_ptr<WCHAR[]> buf(new WCHAR[kMaxPath]);
  DWORD res = GetFinalPathNameByHandleW(h, buf.get(), kMaxPath, 0);
  if (res > 0 && res < kMaxPath) {
    *result = std::move(buf);
    return true;
  } else {
    return false;
  }
}

bool ReadDirectorySymlink(const blaze_util::Path& name, string* result) {
  unique_ptr<WCHAR[]> result_ptr;
  if (!RealPath(name.AsNativePath().c_str(), &result_ptr)) {
    return false;
  }
  *result = WstringToCstring(RemoveUncPrefixMaybe(result_ptr.get()));
  return true;
}

bool PathExists(const string& path) { return PathExists(Path(path)); }

bool PathExists(const Path& path) {
  if (path.IsEmpty()) {
    return false;
  }
  if (path.IsNull()) {
    return true;
  }
  return RealPath(path.AsNativePath().c_str(), nullptr);
}

string MakeCanonical(const char* path) {
  if (IsDevNull(path)) {
    return "NUL";
  }
  if (path == nullptr || path[0] == 0) {
    return "";
  }

  std::wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeCanonical(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
  }

  std::unique_ptr<WCHAR[]> long_realpath;
  if (!RealPath(wpath.c_str(), &long_realpath)) {
    return "";
  }

  // Convert the path to lower-case.
  size_t size =
      wcslen(long_realpath.get()) - (HasUncPrefix(long_realpath.get()) ? 4 : 0);
  unique_ptr<WCHAR[]> lcase_realpath(new WCHAR[size + 1]);
  const WCHAR* p_from = RemoveUncPrefixMaybe(long_realpath.get());
  WCHAR* p_to = lcase_realpath.get();
  while (size-- > 0) {
    *p_to++ = towlower(*p_from++);
  }
  *p_to = 0;
  return WstringToCstring(lcase_realpath.get());
}

static bool CanReadFileW(const wstring& path) {
  AutoHandle handle(CreateFileW(path.c_str(), GENERIC_READ, kAllShare, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
  return handle.IsValid();
}

bool CanReadFile(const std::string& path) {
  return CanReadFile(Path(path));
}

bool CanReadFile(const Path& path) {
  return CanReadFileW(path.AsNativePath());
}

bool CanExecuteFile(const std::string& path) {
  return CanExecuteFile(Path(path));
}

bool CanExecuteFile(const Path& path) {
  std::wstring p = path.AsNativePath();
  if (p.size() < 4) {
    return false;
  }
  std::wstring ext = p.substr(p.size() - 4);
  return CanReadFileW(p) &&
         (ext == L".exe" || ext == L".com" || ext == L".cmd" || ext == L".bat");
}

bool CanAccessDirectory(const std::string& path) {
  return CanAccessDirectory(Path(path));
}

bool CanAccessDirectory(const Path& path) {
  DWORD attr = ::GetFileAttributesW(path.AsNativePath().c_str());
  if ((attr == INVALID_FILE_ATTRIBUTES) || !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
    // The path doesn't exist or is not a directory.
    return false;
  }

  // The only easy way to know if a directory is writable is by attempting to
  // open a file for writing in it.
  // File name with Thread ID avoids races among concurrent Bazel processes.
  std::string dummy_name = "bazel_directory_access_test_";
  dummy_name += std::to_string(::GetCurrentThreadId());

  Path dummy_path = path.GetRelative(dummy_name);

  // Attempt to open the dummy file for read/write access.
  // If the file happens to exist, no big deal, we won't overwrite it thanks to
  // OPEN_ALWAYS.
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ dummy_path.AsNativePath().c_str(),
      /* dwDesiredAccess */ GENERIC_WRITE | GENERIC_READ,
      /* dwShareMode */ kAllShare,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_ALWAYS,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  DWORD err = GetLastError();
  if (handle == INVALID_HANDLE_VALUE) {
    // We couldn't open the file, and not because the dummy file already exists.
    // Consequently it is because `path` doesn't exist.
    return false;
  }
  // The fact that we could open the file, regardless of it existing beforehand
  // or not, means the directory also exists and we can read/write in it.
  CloseHandle(handle);
  if (err != ERROR_ALREADY_EXISTS) {
    // The file didn't exist before, but due to OPEN_ALWAYS we created it just
    // now, so do delete it.
    ::DeleteFileW(dummy_path.AsNativePath().c_str());
  }  // Otherwise the file existed before, leave it alone.
  return true;
}

bool IsDirectoryW(const wstring& path) {
  // Attempt opening the path, which may be anything -- a file, a directory, a
  // symlink, even a dangling symlink is fine.
  // Follow reparse points in order to return false for dangling ones.
  AutoHandle h(CreateFileW(path.c_str(), 0, kAllShare, NULL, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, NULL));
  BY_HANDLE_FILE_INFORMATION info;
  return h.IsValid() && GetFileInformationByHandle(h, &info) &&
         info.dwFileAttributes != INVALID_FILE_ATTRIBUTES &&
         (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}

bool IsDirectory(const string& path) { return IsDirectory(Path(path)); }

bool IsDirectory(const Path& path) {
  if (path.IsEmpty() || path.IsNull()) {
    return false;
  }
  return IsDirectoryW(path.AsNativePath());
}

void SyncFile(const string& path) {
  // No-op on Windows native; unsupported by Cygwin.
  // fsync always fails on Cygwin with "Permission denied" for some reason.
}

void SyncFile(const Path& path) {}

bool MakeDirectoriesW(const wstring& path, unsigned int mode) {
  if (path.empty()) {
    return false;
  }
  std::wstring abs_path;
  std::string error;
  if (!AsAbsoluteWindowsPath(path, &abs_path, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeDirectoriesW(" << blaze_util::WstringToCstring(path)
        << "): " << error;
  }
  if (IsRootDirectoryW(abs_path) || IsDirectoryW(abs_path)) {
    return true;
  }
  wstring parent = SplitPathW(abs_path).first;
  if (parent.empty()) {
    // Since `abs_path` is not a root directory, there should have been at least
    // one directory above it.
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeDirectoriesW(" << blaze_util::WstringToCstring(abs_path)
        << ") could not find dirname: " << GetLastErrorString();
  }
  return MakeDirectoriesW(parent, mode) &&
         ::CreateDirectoryW(abs_path.c_str(), NULL) == TRUE;
}

bool MakeDirectories(const string& path, unsigned int mode) {
  // TODO(laszlocsomor): respect `mode` to the extent that it's possible on
  // Windows; it's currently ignored.
  if (path.empty() || IsDevNull(path.c_str())) {
    return false;
  }
  wstring wpath;
  // According to MSDN, CreateDirectory's limit without the UNC prefix is
  // 248 characters (so it could fit another filename before reaching MAX_PATH).
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeDirectories(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return MakeDirectoriesW(wpath, mode);
}

bool MakeDirectories(const Path& path, unsigned int mode) {
  return MakeDirectoriesW(path.AsNativePath(), mode);
}

string CreateTempDir(const std::string &prefix) {
  string result = prefix + blaze_util::ToString(GetCurrentProcessId());
  if (!blaze_util::MakeDirectories(result, 0777)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "couldn't create '" << result
        << "': " << blaze_util::GetLastErrorString();
  }
  return result;
}

static bool RemoveContents(wstring path) {
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");

  if (path.find(L"\\\\?\\") != 0) {
    path = wstring(L"\\\\?\\") + path;
  }
  if (path.back() != '\\') {
    path.push_back('\\');
  }

  WIN32_FIND_DATAW metadata;
  HANDLE handle = FindFirstFileW((path + L"*").c_str(), &metadata);
  if (handle == INVALID_HANDLE_VALUE) {
    return true;  // directory doesn't exist
  }

  bool result = true;
  do {
    wstring childname = metadata.cFileName;
    if (kDot != childname && kDotDot != childname) {
      wstring childpath = path + childname;
      if ((metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        // If this is not a junction, delete its contents recursively.
        // Finally delete this directory/junction too.
        if (((metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0 &&
             !RemoveContents(childpath)) ||
            !::RemoveDirectoryW(childpath.c_str())) {
          result = false;
          break;
        }
      } else {
        if (!::DeleteFileW(childpath.c_str())) {
          result = false;
          break;
        }
      }
    }
  } while (FindNextFileW(handle, &metadata));
  FindClose(handle);
  return result;
}

static bool RemoveRecursivelyW(const wstring& path) {
  DWORD attrs = ::GetFileAttributesW(path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    // Path does not exist.
    return true;
  }
  if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
    if (!(attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
      // Path is a directory; unlink(2) also cannot remove directories.
      return RemoveContents(path) && ::RemoveDirectoryW(path.c_str());
    }
    // Otherwise it's a junction, remove using RemoveDirectoryW.
    return ::RemoveDirectoryW(path.c_str()) == TRUE;
  } else {
    // Otherwise it's a file, remove using DeleteFileW.
    return ::DeleteFileW(path.c_str()) == TRUE;
  }
}

bool RemoveRecursively(const string& path) {
  return RemoveRecursivelyW(Path(path).AsNativePath());
}

static inline void ToLowerW(WCHAR* p) {
  while (*p) {
    *p++ = towlower(*p);
  }
}

std::wstring GetCwdW() {
  static constexpr size_t kBufSmall = MAX_PATH;
  WCHAR buf[kBufSmall];
  DWORD len = GetCurrentDirectoryW(kBufSmall, buf);
  if (len == 0) {
    DWORD err = GetLastError();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "GetCurrentDirectoryW failed (error " << err << ")";
  }

  if (len < kBufSmall) {
    ToLowerW(buf);
    return std::wstring(buf);
  }

  unique_ptr<WCHAR[]> buf_big(new WCHAR[len]);
  len = GetCurrentDirectoryW(len, buf_big.get());
  if (len == 0) {
    DWORD err = GetLastError();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "GetCurrentDirectoryW failed (error " << err << ")";
  }
  ToLowerW(buf_big.get());
  return std::wstring(buf_big.get());
}

string GetCwd() {
  return WstringToCstring(RemoveUncPrefixMaybe(GetCwdW().c_str()));
}

bool ChangeDirectory(const string& path) {
  string spath;
  string error;
  if (!AsShortWindowsPath(path, &spath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ChangeDirectory(" << path << "): failed: " << error;
  }
  return ::SetCurrentDirectoryA(spath.c_str()) == TRUE;
}

class DirectoryTreeWalkerW : public DirectoryEntryConsumerW {
 public:
  DirectoryTreeWalkerW(vector<wstring>* files,
                       _ForEachDirectoryEntryW walk_entries)
      : _files(files), _walk_entries(walk_entries) {}

  void Consume(const wstring& path, bool follow_directory) override {
    if (follow_directory) {
      Walk(path);
    } else {
      _files->push_back(path);
    }
  }

  void Walk(const wstring& path) { _walk_entries(path, this); }

 private:
  vector<wstring>* _files;
  _ForEachDirectoryEntryW _walk_entries;
};

void ForEachDirectoryEntryW(const wstring& path,
                            DirectoryEntryConsumerW* consume) {
  wstring wpath;
  if (path.empty() || IsDevNull(path.c_str())) {
    return;
  }
  string error;
  if (!AsWindowsPath(path, &wpath, &error)) {
    std::string err = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ForEachDirectoryEntryW(" << WstringToCstring(path)
        << "): AsWindowsPath failed: " << err;
  }

  static const wstring kUncPrefix(L"\\\\?\\");
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");
  // Always add an UNC prefix to ensure we can work with long paths.
  if (!HasUncPrefix(wpath.c_str())) {
    wpath = kUncPrefix + wpath;
  }
  // Unconditionally add a trailing backslash. We know `wpath` has no trailing
  // backslash because it comes from AsWindowsPath whose output is always
  // normalized (see NormalizeWindowsPath).
  wpath.append(L"\\");
  WIN32_FIND_DATAW metadata;
  HANDLE handle = ::FindFirstFileW((wpath + L"*").c_str(), &metadata);
  if (handle == INVALID_HANDLE_VALUE) {
    return;  // directory does not exist or is empty
  }

  do {
    if (kDot != metadata.cFileName && kDotDot != metadata.cFileName) {
      wstring wname = wpath + metadata.cFileName;
      wstring name(/* omit prefix */ 4 + wname.c_str());
      bool is_dir = (metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
      bool is_junc =
          (metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
      consume->Consume(name, is_dir && !is_junc);
    }
  } while (::FindNextFileW(handle, &metadata));
  ::FindClose(handle);
}

void GetAllFilesUnderW(const wstring& path, vector<wstring>* result) {
  _GetAllFilesUnderW(path, result, &ForEachDirectoryEntryW);
}

void _GetAllFilesUnderW(const wstring& path, vector<wstring>* result,
                        _ForEachDirectoryEntryW walk_entries) {
  DirectoryTreeWalkerW(result, walk_entries).Walk(path);
}

void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  wstring wpath;
  if (path.empty() || IsDevNull(path.c_str())) {
    return;
  }
  string error;
  if (!AsWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ForEachDirectoryEntry(" << path
        << "): AsWindowsPath failed: " << GetLastErrorString();
  }

  static const wstring kUncPrefix(L"\\\\?\\");
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");
  // Always add an UNC prefix to ensure we can work with long paths.
  if (!HasUncPrefix(wpath.c_str())) {
    wpath = kUncPrefix + wpath;
  }
  // Unconditionally add a trailing backslash. We know `wpath` has no trailing
  // backslash because it comes from AsWindowsPath whose output is always
  // normalized (see NormalizeWindowsPath).
  wpath.append(L"\\");
  WIN32_FIND_DATAW metadata;
  HANDLE handle = ::FindFirstFileW((wpath + L"*").c_str(), &metadata);
  if (handle == INVALID_HANDLE_VALUE) {
    return;  // directory does not exist or is empty
  }

  do {
    if (kDot != metadata.cFileName && kDotDot != metadata.cFileName) {
      wstring wname = wpath + metadata.cFileName;
      string name(WstringToCstring(/* omit prefix */ 4 + wname.c_str()));
      bool is_dir = (metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
      bool is_junc =
          (metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
      consume->Consume(name, is_dir && !is_junc);
    }
  } while (::FindNextFileW(handle, &metadata));
  ::FindClose(handle);
}

}  // namespace blaze_util
