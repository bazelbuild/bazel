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
#include <ctype.h>  // isalpha
#include <wchar.h>  // wcslen
#include <wctype.h>  // iswalpha
#include <windows.h>

#include <memory>  // unique_ptr
#include <sstream>
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
using bazel::windows::OpenDirectory;
using std::basic_string;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;
using std::wstring;

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

  bool IsUntampered(const string& path) override;
  bool SetToNow(const string& path) override;
  bool SetToDistantFuture(const string& path) override;

 private:
  // 9 years in the future.
  const FILETIME near_future_;
  // 10 years in the future.
  const FILETIME distant_future_;

  static FILETIME GetNow();
  static FILETIME GetFuture(WORD years);
  static bool Set(const string& path, FILETIME time);
};

bool WindowsFileMtime::IsUntampered(const string& path) {
  if (path.empty() || IsDevNull(path.c_str())) {
    return false;
  }

  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WindowsFileMtime::IsUntampered(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
  }

  // Get attributes, to check if the file exists. (It may still be a dangling
  // junction.)
  DWORD attrs = GetFileAttributesW(wpath.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return false;
  }

  bool is_directory = attrs & FILE_ATTRIBUTE_DIRECTORY;
  AutoHandle handle(CreateFileW(
      /* lpFileName */ wpath.c_str(),
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

bool WindowsFileMtime::SetToNow(const string& path) {
  return Set(path, GetNow());
}

bool WindowsFileMtime::SetToDistantFuture(const string& path) {
  return Set(path, distant_future_);
}

bool WindowsFileMtime::Set(const string& path, FILETIME time) {
  if (path.empty()) {
    return false;
  }
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WindowsFileMtime::Set(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }

  AutoHandle handle(::CreateFileW(
      /* lpFileName */ wpath.c_str(),
      /* dwDesiredAccess */ FILE_WRITE_ATTRIBUTES,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */
      IsDirectoryW(wpath)
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

static bool OpenFileForReading(const string& filename, HANDLE* result) {
  if (filename.empty()) {
    return false;
  }
  // TODO(laszlocsomor): remove the following check; it won't allow opening NUL.
  if (IsDevNull(filename.c_str())) {
    return true;
  }
  wstring wfilename;
  string error;
  if (!AsAbsoluteWindowsPath(filename, &wfilename, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "OpenFileForReading(" << filename
        << "): AsAbsoluteWindowsPath failed: " << error;
  }
  *result = ::CreateFileW(
      /* lpFileName */ wfilename.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
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
  HANDLE handle;
  if (!OpenFileForReading(filename, &handle)) {
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
  if (IsDevNull(filename.c_str())) {
    // mimic read(2) behavior: we can always read 0 bytes from /dev/null
    return true;
  }
  HANDLE handle;
  if (!OpenFileForReading(filename, &handle)) {
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
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(filename, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WriteFile(" << filename
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }

  UnlinkPathW(wpath);  // We don't care about the success of this.
  AutoHandle handle(::CreateFileW(
      /* lpFileName */ wpath.c_str(),
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

  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(file_path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "UnlinkPath(" << file_path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return UnlinkPathW(wpath);
}

class JunctionResolver {
 public:
  JunctionResolver();

  // Resolves junctions, or simply checks file existence (if not a junction).
  //
  // Returns true if `path` is not a junction and it exists.
  // Returns true if `path` is a junction and can be successfully resolved and
  // its target exists.
  // Returns false otherwise.
  //
  // If `result` is not nullptr and the method returned false, then this will be
  // reset to point to a new WCHAR buffer containing the final resolved path.
  // If `path` was a junction, this will be the fully resolved path, otherwise
  // it will be a copy of `path`.
  bool Resolve(const WCHAR* path, std::unique_ptr<WCHAR[]>* result);

 private:
  static const int kMaximumJunctionDepth;

  // This struct is a simplified version of REPARSE_DATA_BUFFER, defined by
  // the <Ntifs.h> header file, which is not available on some systems.
  // This struct removes the original one's union keeping only
  // MountPointReparseBuffer, while also renames some fields to reflect how
  // ::DeviceIoControl actually uses them when reading junction data.
  typedef struct _ReparseMountPointData {
    static const int kSize = MAXIMUM_REPARSE_DATA_BUFFER_SIZE;

    ULONG ReparseTag;
    USHORT Dummy1;
    USHORT Dummy2;
    USHORT Dummy3;
    USHORT Dummy4;
    // Length of string in PathBuffer, in WCHARs, including the "\??\" prefix
    // and the null-terminator.
    //
    // Reparse points use the "\??\" prefix instead of "\\?\", presumably
    // because the junction is resolved by the kernel and it points to a Device
    // Object path (which is what the kernel understands), and "\??" is a device
    // path. ("\??" is shorthand for "\DosDevices" under which disk drives
    // reside, e.g. "C:" is a symlink to "\DosDevices\C:" aka "\??\C:").
    // See (on 2017-01-04):
    // https://msdn.microsoft.com/en-us/library/windows/hardware/ff565384(v=vs.85).aspx
    // https://msdn.microsoft.com/en-us/library/windows/hardware/ff557762(v=vs.85).aspx
    USHORT Size;
    USHORT Dummy5;
    // First character of the string returned by ::DeviceIoControl. The rest of
    // the string follows this in memory, that's why the caller must allocate
    // kSize bytes and cast that data to ReparseMountPointData.
    WCHAR PathBuffer[1];
  } ReparseMountPointData;

  uint8_t reparse_buffer_bytes_[ReparseMountPointData::kSize];
  ReparseMountPointData* reparse_buffer_;

  bool Resolve(const WCHAR* path, std::unique_ptr<WCHAR[]>* result,
               int max_junction_depth);
};

// Maximum reparse point depth on Windows 8 and above is 63.
// Source (on 2016-12-20):
// https://msdn.microsoft.com/en-us/library/windows/desktop/aa365503(v=vs.85).aspx
const int JunctionResolver::kMaximumJunctionDepth = 63;

JunctionResolver::JunctionResolver()
    : reparse_buffer_(
          reinterpret_cast<ReparseMountPointData*>(reparse_buffer_bytes_)) {
  reparse_buffer_->ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
}

bool JunctionResolver::Resolve(const WCHAR* path, unique_ptr<WCHAR[]>* result,
                               int max_junction_depth) {
  DWORD attributes = ::GetFileAttributesW(path);
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    // `path` does not exist.
    return false;
  } else {
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      // `path` is a junction. GetFileAttributesW succeeds for these even if
      // their target does not exist. We need to resolve the target and check if
      // that exists. (There seems to be no API function for this.)
      if (max_junction_depth <= 0) {
        // Too many levels of junctions. Simply say this file doesn't exist.
        return false;
      }
      // Get a handle to the directory.
      AutoHandle handle(OpenDirectory(path, /* read_write */ false));
      if (!handle.IsValid()) {
        // Opening the junction failed for whatever reason. For all intents and
        // purposes we can treat this file as if it didn't exist.
        return false;
      }
      // Read out the junction data.
      DWORD bytes_returned;
      BOOL ok = ::DeviceIoControl(
          handle, FSCTL_GET_REPARSE_POINT, NULL, 0, reparse_buffer_,
          MAXIMUM_REPARSE_DATA_BUFFER_SIZE, &bytes_returned, NULL);
      if (!ok) {
        // Reading the junction data failed. For all intents and purposes we can
        // treat this file as if it didn't exist.
        return false;
      }
      reparse_buffer_->PathBuffer[reparse_buffer_->Size - 1] = UNICODE_NULL;
      // Check if the junction target exists.
      return Resolve(reparse_buffer_->PathBuffer, result,
                     max_junction_depth - 1);
    }
  }
  // `path` is a normal file or directory.
  if (result) {
    size_t len = wcslen(path) + 1;
    result->reset(new WCHAR[len]);
    memcpy(result->get(), path, len * sizeof(WCHAR));
  }
  return true;
}

bool JunctionResolver::Resolve(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  return Resolve(path, result, kMaximumJunctionDepth);
}

class SymlinkResolver {
 public:
  SymlinkResolver();

  // Resolves symlink to its actual path.
  //
  // Returns true if `path` is not a symlink and it exists.
  // Returns true if `path` is a symlink and can be successfully resolved.
  // Returns false otherwise.
  //
  // If `result` is not nullptr and the method returned true, then this will be
  // reset to point to a new WCHAR buffer containing the resolved path.
  // If `path` is a symlink, this will be the resolved path, otherwise
  // it will be a copy of `path`.
  bool Resolve(const WCHAR* path, std::unique_ptr<WCHAR[]>* result);

 private:
  // Symbolic Link Reparse Data Buffer is described at:
  // https://msdn.microsoft.com/en-us/library/cc232006.aspx
  typedef struct _ReparseSymbolicLinkData {
    static const int kSize = MAXIMUM_REPARSE_DATA_BUFFER_SIZE;
    ULONG ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    USHORT SubstituteNameOffset;
    USHORT SubstituteNameLength;
    USHORT PrintNameOffset;
    USHORT PrintNameLength;
    ULONG Flags;
    WCHAR PathBuffer[1];
  } ReparseSymbolicLinkData;

  uint8_t reparse_buffer_bytes_[ReparseSymbolicLinkData::kSize];
  ReparseSymbolicLinkData* reparse_buffer_;
};

SymlinkResolver::SymlinkResolver()
    : reparse_buffer_(
          reinterpret_cast<ReparseSymbolicLinkData*>(reparse_buffer_bytes_)) {}

bool SymlinkResolver::Resolve(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  DWORD attributes = ::GetFileAttributesW(path);
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    // `path` does not exist.
    return false;
  } else {
    if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      bool is_dir = attributes & FILE_ATTRIBUTE_DIRECTORY;
      AutoHandle handle(
          CreateFileW(path, FILE_READ_EA,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      NULL, OPEN_EXISTING,
                      (is_dir ? FILE_FLAG_BACKUP_SEMANTICS : 0) |
                          FILE_FLAG_OPEN_REPARSE_POINT,
                      NULL));
      if (!handle.IsValid()) {
        // Opening the symlink failed for whatever reason. For all intents and
        // purposes we can treat this file as if it didn't exist.
        return false;
      }
      // Read out the reparse point data.
      DWORD bytes_returned;
      BOOL ok = ::DeviceIoControl(
          handle, FSCTL_GET_REPARSE_POINT, NULL, 0, reparse_buffer_,
          MAXIMUM_REPARSE_DATA_BUFFER_SIZE, &bytes_returned, NULL);
      if (!ok) {
        // Reading the symlink data failed. For all intents and purposes we can
        // treat this file as if it didn't exist.
        return false;
      }
      if (reparse_buffer_->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
        if (result) {
          size_t len = reparse_buffer_->SubstituteNameLength / sizeof(WCHAR);
          result->reset(new WCHAR[len + 1]);
          const WCHAR* substituteName =
              reparse_buffer_->PathBuffer +
              (reparse_buffer_->SubstituteNameOffset / sizeof(WCHAR));
          wcsncpy_s(result->get(), len + 1, substituteName, len);
          result->get()[len] = UNICODE_NULL;
        }
        return true;
      }
    }
  }
  // `path` is a normal file or directory.
  if (result) {
    size_t len = wcslen(path) + 1;
    result->reset(new WCHAR[len]);
    memcpy(result->get(), path, len * sizeof(WCHAR));
  }
  return true;
}

bool ReadDirectorySymlink(const string& name, string* result) {
  wstring wname;
  string error;
  if (!AsAbsoluteWindowsPath(name, &wname, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ReadDirectorySymlink(" << name
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  unique_ptr<WCHAR[]> result_ptr;
  if (!JunctionResolver().Resolve(wname.c_str(), &result_ptr)) {
    return false;
  }
  *result = WstringToCstring(RemoveUncPrefixMaybe(result_ptr.get())).get();
  return true;
}

bool ReadSymlinkW(const wstring& name, wstring* result) {
  wstring wname;
  string error;
  if (!AsAbsoluteWindowsPath(name, &wname, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ReadSymlinkW(" << name
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  unique_ptr<WCHAR[]> result_ptr;
  if (!SymlinkResolver().Resolve(wname.c_str(), &result_ptr)) {
    return false;
  }
  *result = RemoveUncPrefixMaybe(result_ptr.get());
  return true;
}

bool PathExists(const string& path) {
  if (path.empty()) {
    return false;
  }
  if (IsDevNull(path.c_str())) {
    return true;
  }
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "PathExists(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return JunctionResolver().Resolve(wpath.c_str(), nullptr);
}

bool MakeCanonical(const std::string& path, std::string* result,
                   std::string* error) {
  if (path.empty()) {
    result->clear();
    return true;
  }
  if (IsDevNull(path.c_str())) {
    *result = "NUL";
    return true;
  }
  wstring wpath;
  if (!AsAbsoluteWindowsPath(path, &wpath, error)) {
    return false;
  }

  // Resolve all segments of the path. Do this from leaf to root, so we always
  // know that the path's tail is resolved and junctions may be found only in
  // its head.
  std::vector<wstring> realpath_reversed;
  while (true) {
    // Resolve the last segment.
    unique_ptr<WCHAR[]> realpath;
    if (!JunctionResolver().Resolve(wpath.c_str(), &realpath)) {
      // The path doesn't exist or there are too many levels of indirection,
      // so just give up.
      if (error) {
        *error = "Cannot canonicalize '" + path +
                 "': too many indirections or path does not exist.";
      }
      return false;
    }
    // The last segment is surely not a junction anymore. Split it off the path
    // and keep resolving its ancestors until we reach the root directory.
    pair<wstring, wstring> split(SplitPathW(realpath.get()));
    if (split.second.empty()) {
      // `wpath` was a root directory, we're done.
      realpath_reversed.push_back(split.first);
      break;
    } else {
      // `wpath` was not yet a root directory, split off the last segment and
      // store it in the stack, keep resolving the rest.
      realpath_reversed.push_back(split.second);
      wpath = split.first;
    }
  }

  // Concatenate the segments in reverse order.
  int segment_cnt = 0;
  std::wstringstream builder;
  for (auto segment = realpath_reversed.crbegin();
       segment != realpath_reversed.crend(); ++segment) {
    if (segment_cnt < 2) {
      segment_cnt++;
    } else {
      // Start appending '\' separator after not the first but the second
      // segment, since the first segment is a drive name "c:\" and already
      // has the separator.
      builder << L"\\";
    }
    builder << *segment;
  }
  wstring realpath(builder.str());
  if (HasUncPrefix(realpath.c_str())) {
    // `realpath` has an UNC prefix if `path` did, or if `path` contained
    // junctions.
    // In the first case, the UNC prefix is the usual "\\?\", but in the second
    // case it is "\??\", because that's what the junction resolution yields,
    // because that's the prefix the filesystem uses for storing junction
    // values.
    // Since "\??\" is only meaningful for the kernel and not for usermode
    // Win32 API functions, we need to replace this prefix with the usual "\\?\"
    // one.
    realpath[1] = L'\\';
  }

  // Resolve all 8dot3 style segments of the path, if any. The input path may
  // have had some. Junctions may also refer to 8dot3 names.
  unique_ptr<WCHAR[]> long_realpath;
  wstring werror(GetLongPath(realpath.c_str(), &long_realpath));
  if (!werror.empty()) {
    if (error) {
      *error = WstringToCstring(werror.c_str()).get();
    }
    return false;
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
  *result = WstringToCstring(lcase_realpath.get()).get();
  return true;
}

static bool CanReadFileW(const wstring& path) {
  DWORD attrs = ::GetFileAttributesW(path.c_str());
  if ((attrs == INVALID_FILE_ATTRIBUTES) ||
      (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
    // The path doesn't exist or is a directory/junction.
    return false;
  }
  // The only easy way to find out if a file is readable is to attempt to open
  // it for reading.
  AutoHandle handle(::CreateFileW(
      /* lpFileName */ path.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));
  return handle.IsValid();
}

bool CanReadFile(const std::string& path) {
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "CanReadFile(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return CanReadFileW(wpath);
}

bool CanExecuteFile(const std::string& path) {
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "CanExecuteFile(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return CanReadFileW(wpath) && (ends_with(wpath, wstring(L".exe")) ||
                                 ends_with(wpath, wstring(L".com")) ||
                                 ends_with(wpath, wstring(L".cmd")) ||
                                 ends_with(wpath, wstring(L".bat")));
}

bool CanAccessDirectory(const std::string& path) {
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "CanAccessDirectory(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  DWORD attr = ::GetFileAttributesW(wpath.c_str());
  if ((attr == INVALID_FILE_ATTRIBUTES) || !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
    // The path doesn't exist or is not a directory.
    return false;
  }

  // The only easy way to know if a directory is writable is by attempting to
  // open a file for writing in it.
  wstring dummy_path = wpath + L"\\bazel_directory_access_test";

  // The path may have just became too long for MAX_PATH, so add the UNC prefix
  // if necessary.
  AddUncPrefixMaybe(&dummy_path);

  // Attempt to open the dummy file for read/write access.
  // If the file happens to exist, no big deal, we won't overwrite it thanks to
  // OPEN_ALWAYS.
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ dummy_path.c_str(),
      /* dwDesiredAccess */ GENERIC_WRITE | GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ | FILE_SHARE_WRITE,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_ALWAYS,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  DWORD err = GetLastError();
  if (handle == INVALID_HANDLE_VALUE && err != ERROR_ALREADY_EXISTS) {
    // We couldn't open the file, and not because the dummy file already exists.
    // Consequently it is because `wpath` doesn't exist.
    return false;
  }
  // The fact that we could open the file, regardless of it existing beforehand
  // or not, means the directory also exists and we can read/write in it.
  CloseHandle(handle);
  if (err != ERROR_ALREADY_EXISTS) {
    // The file didn't exist before, but due to OPEN_ALWAYS we created it just
    // now, so do delete it.
    ::DeleteFileW(dummy_path.c_str());
  }  // Otherwise the file existed before, leave it alone.
  return true;
}

bool IsDirectoryW(const wstring& path) {
  DWORD attrs = ::GetFileAttributesW(path.c_str());
  return (attrs != INVALID_FILE_ATTRIBUTES) &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY) &&
         JunctionResolver().Resolve(path.c_str(), nullptr);
}

bool IsDirectory(const string& path) {
  if (path.empty() || IsDevNull(path.c_str())) {
    return false;
  }
  wstring wpath;
  string error;
  if (!AsAbsoluteWindowsPath(path, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "IsDirectory(" << path
        << "): AsAbsoluteWindowsPath failed: " << error;
    return false;
  }
  return IsDirectoryW(wpath);
}

void SyncFile(const string& path) {
  // No-op on Windows native; unsupported by Cygwin.
  // fsync always fails on Cygwin with "Permission denied" for some reason.
}

bool MakeDirectoriesW(const wstring& path, unsigned int mode) {
  if (path.empty()) {
    return false;
  }
  if (IsRootDirectoryW(path) || IsDirectoryW(path)) {
    return true;
  }
  wstring parent = SplitPathW(path).first;
  if (parent.empty()) {
    // Since `path` is not a root directory, there should have been at least one
    // directory above it.
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeDirectoriesW(" << blaze_util::WstringToString(path)
        << ") could not find dirname: " << GetLastErrorString();
  }
  return MakeDirectoriesW(parent, mode) &&
         ::CreateDirectoryW(path.c_str(), NULL) == TRUE;
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
  return string(
      WstringToCstring(RemoveUncPrefixMaybe(GetCwdW().c_str())).get());
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
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ForEachDirectoryEntryW(" << path
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
      string name(WstringToCstring(/* omit prefix */ 4 + wname.c_str()).get());
      bool is_dir = (metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
      bool is_junc =
          (metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
      consume->Consume(name, is_dir && !is_junc);
    }
  } while (::FindNextFileW(handle, &metadata));
  ::FindClose(handle);
}

}  // namespace blaze_util
