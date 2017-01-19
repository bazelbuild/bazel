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
#include "src/main/cpp/util/file_platform.h"

#include <ctype.h>  // isalpha
#include <wctype.h>  // iswalpha
#include <windows.h>

#include <memory>  // unique_ptr
#include <sstream>
#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows_util.h"

namespace blaze_util {

using std::basic_string;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;
using std::wstring;

// Returns the current working directory as a Windows path.
// The result may have a UNC prefix.
static unique_ptr<WCHAR[]> GetCwdW();

static bool IsDevNull(const string& path);

// Returns true if `path` refers to a directory or (non-dangling) junction.
// `path` must be a normalized Windows path, with UNC prefix (and absolute) if
// necessary.
static bool IsDirectoryW(const wstring& path);

// Returns true the file or junction at `path` is successfully deleted.
// Returns false otherwise, or if `path` doesn't exist or is a directory.
// `path` must be a normalized Windows path, with UNC prefix (and absolute) if
// necessary.
static bool UnlinkPathW(const wstring& path);

static bool IsRootDirectoryW(const wstring& path);

static bool MakeDirectoriesW(const wstring& path);

static bool CanReadFileW(const wstring& path);

// Returns a normalized form of the input `path`.
//
// `path` must be a relative or absolute Windows path, it may use "/" instead of
// "\" but must not be an absolute MSYS path.
// The result won't have a UNC prefix, even if `path` did.
//
// Normalization means removing "." references, resolving ".." references, and
// deduplicating "/" characters while converting them to "\".
// For example if `path` is "foo/../bar/.//qux", the result is "bar\qux".
//
// Uplevel references that cannot go any higher in the directory tree are simply
// ignored, e.g. "c:/.." is normalized to "c:\" and "../../foo" is normalized to
// "foo".
//
// Visible for testing, would be static otherwise.
string NormalizeWindowsPath(string path);

template <typename char_type>
struct CharTraits {
  static bool IsAlpha(char_type ch);
};

template <>
struct CharTraits<char> {
  static bool IsAlpha(char ch) { return isalpha(ch); }
};

template <>
struct CharTraits<wchar_t> {
  static bool IsAlpha(wchar_t ch) { return iswalpha(ch); }
};

template <typename char_type>
static bool IsPathSeparator(char_type ch) {
  return ch == '/' || ch == '\\';
}

template <typename char_type>
static bool HasDriveSpecifierPrefix(const char_type* ch) {
  return CharTraits<char_type>::IsAlpha(ch[0]) && ch[1] == ':';
}

template <typename char_type>
static bool HasUncPrefix(const char_type* path) {
  return path[0] == '\\' && (path[1] == '\\' || path[1] == '?') &&
         (path[2] == '.' || path[2] == '?') && path[3] == '\\';
}

static void AddUncPrefixMaybe(wstring* path) {
  if (path->size() > MAX_PATH && !HasUncPrefix(path->c_str())) {
    *path = wstring(L"\\\\?\\") + *path;
  }
}

class WindowsPipe : public IPipe {
 public:
  WindowsPipe(const HANDLE& read_handle, const HANDLE& write_handle)
      : _read_handle(read_handle), _write_handle(write_handle) {}

  WindowsPipe() = delete;

  virtual ~WindowsPipe() {
    if (_read_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(_read_handle);
      _read_handle = INVALID_HANDLE_VALUE;
    }
    if (_write_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(_write_handle);
      _write_handle = INVALID_HANDLE_VALUE;
    }
  }

  bool Send(const void* buffer, int size) override {
    DWORD actually_written = 0;
    return ::WriteFile(_write_handle, buffer, size, &actually_written, NULL) ==
           TRUE;
  }

  int Receive(void* buffer, int size) override {
    DWORD actually_read = 0;
    return ::ReadFile(_read_handle, buffer, size, &actually_read, NULL)
               ? actually_read
               : -1;
  }

 private:
  HANDLE _read_handle;
  HANDLE _write_handle;
};

IPipe* CreatePipe() {
  // The pipe HANDLEs can be inherited.
  SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
  HANDLE read_handle = INVALID_HANDLE_VALUE;
  HANDLE write_handle = INVALID_HANDLE_VALUE;
  if (!CreatePipe(&read_handle, &write_handle, &sa, 0)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "CreatePipe failed, err=%d", GetLastError());
  }
  return new WindowsPipe(read_handle, write_handle);
}

class WindowsFileMtime : public IFileMtime {
 public:
  WindowsFileMtime()
      : near_future_(GetFuture(9)), distant_future_(GetFuture(10)) {}

  bool GetIfInDistantFuture(const string& path, bool* result) override;
  bool SetToNow(const string& path) override;
  bool SetToDistantFuture(const string& path) override;

 private:
  // 9 years in the future.
  const FILETIME near_future_;
  // 10 years in the future.
  const FILETIME distant_future_;

  static FILETIME GetNow();
  static FILETIME GetFuture(WORD years);
  static bool Set(const string& path, const FILETIME& time);
};

bool WindowsFileMtime::GetIfInDistantFuture(const string& path, bool* result) {
  if (path.empty()) {
    return false;
  }
  if (IsDevNull(path)) {
    *result = false;
    return true;
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }

  windows_util::AutoHandle handle(::CreateFileW(
      /* lpFileName */ wpath.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ IsDirectoryW(wpath)
          ? (FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
          : FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));
  if (handle.handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  FILETIME mtime;
  if (!::GetFileTime(
          /* hFile */ handle.handle,
          /* lpCreationTime */ NULL,
          /* lpLastAccessTime */ NULL,
          /* lpLastWriteTime */ &mtime)) {
    return false;
  }

  // Compare the mtime with `near_future_`, not with `GetNow()` or
  // `distant_future_`.
  // This way we don't need to call GetNow() every time we want to compare (and
  // thus convert a SYSTEMTIME to FILETIME), and we also don't need to worry
  // about potentially unreliable FILETIME equality check (in case it uses
  // floats or something crazy).
  *result = CompareFileTime(&near_future_, &mtime) == -1;
  return true;
}

bool WindowsFileMtime::SetToNow(const string& path) {
  return Set(path, GetNow());
}

bool WindowsFileMtime::SetToDistantFuture(const string& path) {
  return Set(path, distant_future_);
}

bool WindowsFileMtime::Set(const string& path, const FILETIME& time) {
  if (path.empty()) {
    return false;
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }

  windows_util::AutoHandle handle(::CreateFileW(
      /* lpFileName */ wpath.c_str(),
      /* dwDesiredAccess */ FILE_WRITE_ATTRIBUTES,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ IsDirectoryW(wpath)
          ? (FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
          : FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL));
  if (handle.handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  return ::SetFileTime(
             /* hFile */ handle.handle,
             /* lpCreationTime */ NULL,
             /* lpLastAccessTime */ NULL,
             /* lpLastWriteTime */ &time) == TRUE;
}

FILETIME WindowsFileMtime::GetNow() {
  SYSTEMTIME sys_time;
  ::GetSystemTime(&sys_time);
  FILETIME file_time;
  if (!::SystemTimeToFileTime(&sys_time, &file_time)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "WindowsFileMtime::GetNow: SystemTimeToFileTime failed, err=%d",
         GetLastError());
  }
  return file_time;
}

FILETIME WindowsFileMtime::GetFuture(WORD years) {
  SYSTEMTIME future_time;
  GetSystemTime(&future_time);
  future_time.wYear += years;
  future_time.wMonth = 1;
  future_time.wDayOfWeek = 0;
  future_time.wDay = 1;
  future_time.wHour = 0;
  future_time.wMinute = 0;
  future_time.wSecond = 0;
  future_time.wMilliseconds = 0;
  FILETIME file_time;
  if (!::SystemTimeToFileTime(&future_time, &file_time)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "WindowsFileMtime::GetFuture: SystemTimeToFileTime failed, err=%d",
         GetLastError());
  }
  return file_time;
}

IFileMtime* CreateFileMtime() { return new WindowsFileMtime(); }

// Checks if the path is absolute and/or is a root path.
//
// If `must_be_root` is true, then in addition to being absolute, the path must
// also be just the root part, no other components, e.g. "c:\" is both absolute
// and root, but "c:\foo" is just absolute.
template <typename char_type>
static bool IsRootOrAbsolute(const basic_string<char_type>& path,
                             bool must_be_root) {
  // An absolute path is one that starts with "/", "\", "c:/", "c:\",
  // "\\?\c:\", or rarely "\??\c:\" or "\\.\c:\".
  //
  // It is unclear whether the UNC prefix is just "\\?\" or is "\??\" also
  // valid (in some cases it seems to be, though MSDN doesn't mention it).
  return
      // path is (or starts with) "/" or "\"
      ((must_be_root ? path.size() == 1 : !path.empty()) &&
       IsPathSeparator(path[0])) ||
      // path is (or starts with) "c:/" or "c:\" or similar
      ((must_be_root ? path.size() == 3 : path.size() >= 3) &&
       HasDriveSpecifierPrefix(path.c_str()) && IsPathSeparator(path[2])) ||
      // path is (or starts with) "\\?\c:\" or "\??\c:\" or similar
      ((must_be_root ? path.size() == 7 : path.size() >= 7) &&
       HasUncPrefix(path.c_str()) &&
       HasDriveSpecifierPrefix(path.c_str() + 4) && IsPathSeparator(path[6]));
}

pair<string, string> SplitPath(const string& path) {
  if (path.empty()) {
    return std::make_pair("", "");
  }

  size_t pos = path.size() - 1;
  for (auto it = path.crbegin(); it != path.crend(); ++it, --pos) {
    if (IsPathSeparator(*it)) {
      if ((pos == 2 || pos == 6) && IsRootDirectory(path.substr(0, pos + 1))) {
        // Windows path, top-level directory, e.g. "c:\foo",
        // result is ("c:\", "foo").
        // Or UNC path, top-level directory, e.g. "\\?\c:\foo"
        // result is ("\\?\c:\", "foo").
        return std::make_pair(
            // Include the "/" or "\" in the drive specifier.
            path.substr(0, pos + 1), path.substr(pos + 1));
      } else {
        // Windows path (neither top-level nor drive root), Unix path, or
        // relative path.
        return std::make_pair(
            // If the only "/" is the leading one, then that shall be the first
            // pair element, otherwise the substring up to the rightmost "/".
            pos == 0 ? path.substr(0, 1) : path.substr(0, pos),
            // If the rightmost "/" is the tail, then the second pair element
            // should be empty.
            pos == path.size() - 1 ? "" : path.substr(pos + 1));
      }
    }
  }
  // Handle the case with no '/' or '\' in `path`.
  return std::make_pair("", path);
}

class MsysRoot {
 public:
  static bool IsValid();
  static const string& GetPath();
  static void ResetForTesting() { instance_.initialized_ = false; }

 private:
  bool initialized_;
  bool valid_;
  string path_;
  static MsysRoot instance_;

  static bool Get(string* path);

  MsysRoot() : initialized_(false) {}
  void InitIfNecessary();
};

MsysRoot MsysRoot::instance_;

void ResetMsysRootForTesting() { MsysRoot::ResetForTesting(); }

bool MsysRoot::IsValid() {
  instance_.InitIfNecessary();
  return instance_.valid_;
}

const string& MsysRoot::GetPath() {
  instance_.InitIfNecessary();
  return instance_.path_;
}

bool MsysRoot::Get(string* path) {
  string result;
  char value[MAX_PATH];
  DWORD len = GetEnvironmentVariableA("BAZEL_SH", value, MAX_PATH);
  if (len > 0) {
    result = value;
  } else {
    const char* value2 = getenv("BAZEL_SH");
    if (value2 == nullptr || value2[0] == '\0') {
      PrintError(
          "BAZEL_SH environment variable is not defined, cannot convert MSYS "
          "paths to Windows paths");
      return false;
    }
    result = value2;
  }

  ToLower(&result);

  // BAZEL_SH is usually "c:\tools\msys64\usr\bin\bash.exe", we need to return
  // "c:\tools\msys64". Look for the rightmost msys-looking component.
  while (!IsRootDirectory(result) &&
         Basename(result).find("msys") == string::npos) {
    result = Dirname(result);
  }
  if (IsRootDirectory(result)) {
    return false;
  }
  *path = result;
  return true;
}

void MsysRoot::InitIfNecessary() {
  if (!initialized_) {
    valid_ = Get(&path_);
    initialized_ = true;
  }
}

// Converts a UTF8-encoded `path` to a normalized, widechar Windows path.
//
// Returns true if conversion succeeded and sets the contents of `result` to it.
//
// The `path` may be absolute or relative, and may be a Windows or MSYS path.
// In every case, the output is normalized (see NormalizeWindowsPath).
// The output won't have a UNC prefix, even if `path` did.
//
// Recognizes the drive letter in MSYS paths, so e.g. "/c/windows" becomes
// "c:\windows". Prepends the MSYS root (computed from the BAZEL_SH envvar) to
// absolute MSYS paths, so e.g. "/usr" becomes "c:\tools\msys64\usr".
//
// The result may be longer than MAX_PATH. It's the caller's responsibility to
// prepend the UNC prefix in case they need to pass it to a WinAPI function
// (some require the prefix, some don't), or to quote the path if necessary.
bool AsWindowsPath(const string& path, wstring* result) {
  if (path.empty()) {
    result->clear();
    return true;
  }
  if (IsDevNull(path)) {
    result->assign(L"NUL");
    return true;
  }

  string mutable_path = path;
  if (path[0] == '/') {
    // This is an absolute MSYS path.
    if (path.size() == 2 || (path.size() > 2 && path[2] == '/')) {
      // The path is either "/x" or "/x/" or "/x/something". In all three cases
      // "x" is the drive letter.
      // TODO(laszlocsomor): use GetLogicalDrives to retrieve the list of drives
      // and only apply this heuristic for the valid drives. It's possible that
      // the user has a directory "/a" but no "A:\" drive, so in that case we
      // should prepend the MSYS root.
      mutable_path = path.substr(1, 1) + ":\\";
      if (path.size() > 2) {
        mutable_path += path.substr(3);
      }
    } else {
      // The path is a normal MSYS path e.g. "/usr". Prefix it with the MSYS
      // root.
      if (!MsysRoot::IsValid()) {
        return false;
      }
      mutable_path = JoinPath(MsysRoot::GetPath(), path);
    }
  }  // otherwise this is a relative path, or absolute Windows path.

  result->assign(
      CstringToWstring(NormalizeWindowsPath(mutable_path).c_str()).get());
  return true;
}

bool AsWindowsPathWithUncPrefix(const string& path, wstring* wpath) {
  if (IsDevNull(path)) {
    wpath->assign(L"NUL");
    return true;
  }

  if (!AsWindowsPath(path, wpath)) {
    PrintError("AsWindowsPathWithUncPrefix(%s): AsWindowsPath failed, err=%d\n",
               path.c_str(), GetLastError());
    return false;
  }
  if (!IsAbsolute(path)) {
    wpath->assign(wstring(GetCwdW().get()) + L"\\" + *wpath);
  }
  AddUncPrefixMaybe(wpath);
  return true;
}

bool AsShortWindowsPath(const string& path, string* result) {
  if (IsDevNull(path)) {
    result->assign("NUL");
    return true;
  }

  result->clear();
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  DWORD size = ::GetShortPathNameW(wpath.c_str(), nullptr, 0);
  if (size == 0) {
    return false;
  }

  unique_ptr<WCHAR[]> wshort(new WCHAR[size]);  // size includes null-terminator
  if (size - 1 != ::GetShortPathNameW(wpath.c_str(), wshort.get(), size)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "AsShortWindowsPath(%s): GetShortPathNameW(%S) failed, err=%d",
         path.c_str(), wpath.c_str(), GetLastError());
  }
  // GetShortPathNameW may preserve the UNC prefix in the result, so strip it.
  WCHAR* result_ptr = wshort.get() + (HasUncPrefix(wshort.get()) ? 4 : 0);

  result->assign(WstringToCstring(result_ptr).get());
  ToLower(result);
  return true;
}

bool ReadFile(const string& filename, string* content, int max_size) {
  if (filename.empty()) {
    return false;
  }
  if (IsDevNull(filename)) {
    content->clear();
    return true;
  }
  wstring wfilename;
  if (!AsWindowsPathWithUncPrefix(filename, &wfilename)) {
    return false;
  }
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ wfilename.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  bool result = ReadFrom(
      [handle](void* buf, int len) {
        DWORD actually_read = 0;
        ::ReadFile(handle, buf, len, &actually_read, NULL);
        return actually_read;
      },
      content, max_size);
  CloseHandle(handle);
  return result;
}

bool WriteFile(const void* data, size_t size, const string& filename) {
  if (IsDevNull(filename)) {
    return true;  // mimic write(2) behavior with /dev/null
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(filename, &wpath)) {
    return false;
  }

  UnlinkPathW(wpath);  // We don't care about the success of this.
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ wpath.c_str(),
      /* dwDesiredAccess */ GENERIC_WRITE,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ CREATE_ALWAYS,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  bool result = WriteTo(
      [handle](const void* buf, size_t bufsize) {
        DWORD actually_written = 0;
        ::WriteFile(handle, buf, bufsize, &actually_written, NULL);
        return actually_written;
      },
      data, size);
  CloseHandle(handle);
  return result;
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
  if (IsDevNull(file_path)) {
    return false;
  }

  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(file_path, &wpath)) {
    return false;
  }
  return UnlinkPathW(wpath);
}

HANDLE OpenDirectory(const WCHAR* path, bool read_write) {
  return ::CreateFileW(
      /* lpFileName */ path,
      /* dwDesiredAccess */ read_write ? (GENERIC_READ | GENERIC_WRITE)
                                       : GENERIC_READ,
      /* dwShareMode */ 0,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_FLAG_OPEN_REPARSE_POINT |
          FILE_FLAG_BACKUP_SEMANTICS,
      /* hTemplateFile */ NULL);
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
      HANDLE handle = OpenDirectory(path, /* read_write */ false);
      if (handle == INVALID_HANDLE_VALUE) {
        // Opening the junction failed for whatever reason. For all intents and
        // purposes we can treat this file as if it didn't exist.
        return false;
      }
      // Read out the junction data.
      DWORD bytes_returned;
      BOOL ok = ::DeviceIoControl(
          handle, FSCTL_GET_REPARSE_POINT, NULL, 0, reparse_buffer_,
          MAXIMUM_REPARSE_DATA_BUFFER_SIZE, &bytes_returned, NULL);
      CloseHandle(handle);
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

bool PathExists(const string& path) {
  if (path.empty()) {
    return false;
  }
  if (IsDevNull(path)) {
    return true;
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    PrintError("PathExists(%s): AsWindowsPathWithUncPrefix failed, err=%d\n",
               path.c_str(), GetLastError());
    return false;
  }
  return JunctionResolver().Resolve(wpath.c_str(), nullptr);
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
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ path.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  } else {
    CloseHandle(handle);
    return true;
  }
}

bool CanReadFile(const std::string& path) {
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  return CanReadFileW(wpath);
}

bool CanExecuteFile(const std::string& path) {
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  return CanReadFileW(wpath) && (ends_with(wpath, wstring(L".exe")) ||
                                 ends_with(wpath, wstring(L".com")) ||
                                 ends_with(wpath, wstring(L".cmd")) ||
                                 ends_with(wpath, wstring(L".bat")));
}

bool CanAccessDirectory(const std::string& path) {
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
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

static bool IsDevNull(const string& path) {
  return path == "/dev/null" || AsLower(path) == "nul";
}

static bool IsDirectoryW(const wstring& path) {
  DWORD attrs = ::GetFileAttributesW(path.c_str());
  return (attrs != INVALID_FILE_ATTRIBUTES) &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY) &&
         JunctionResolver().Resolve(path.c_str(), nullptr);
}

bool IsDirectory(const string& path) {
  if (path.empty() || IsDevNull(path)) {
    return false;
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  return IsDirectoryW(wpath);
}

bool IsRootDirectory(const string& path) {
  return IsRootOrAbsolute(path, true);
}

bool IsAbsolute(const string& path) { return IsRootOrAbsolute(path, false); }

void SyncFile(const string& path) {
  // No-op on Windows native; unsupported by Cygwin.
  // fsync always fails on Cygwin with "Permission denied" for some reason.
}

#ifdef COMPILER_MSVC
time_t GetMtimeMillisec(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetMtimeMillisec is not implemented on Windows");
  return -1;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool SetMtimeMillisec(const string& path, time_t mtime) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::SetMtimeMillisec is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

static bool IsRootDirectoryW(const wstring& path) {
  return IsRootOrAbsolute(path, true);
}

static bool MakeDirectoriesW(const wstring& path) {
  if (path.empty()) {
    return false;
  }
  if (IsRootDirectoryW(path) || IsDirectoryW(path)) {
    return true;
  }
  int last_separator = path.rfind(L"\\");
  if (last_separator < 0) {
    // Since `path` is not a root directory, there must be at least one
    // directory above it.
    pdie(255, "MakeDirectoriesW(%S), could not find dirname", path.c_str());
  }
  wstring parent = path.substr(0, last_separator);
  if (!MakeDirectoriesW(parent)) {
    return false;
  }
  if (!::CreateDirectoryW(path.c_str(), nullptr)) {
    PrintError("MakeDirectoriesW(%S), CreateDirectoryW failed, err=%d",
               path.c_str(), GetLastError());
    return false;
  }
  return true;
}

bool MakeDirectories(const string& path, unsigned int mode) {
  // TODO(laszlocsomor): respect `mode` to the extent that it's possible on
  // Windows; it's currently ignored.
  if (path.empty() || IsDevNull(path) || IsRootDirectory(path)) {
    return false;
  }
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  return MakeDirectoriesW(wpath);
}

static unique_ptr<WCHAR[]> GetCwdW() {
  DWORD len = ::GetCurrentDirectoryW(0, nullptr);
  unique_ptr<WCHAR[]> cwd(new WCHAR[len]);
  if (!::GetCurrentDirectoryW(len, cwd.get())) {
    die(255, "GetCurrentDirectoryW failed, err=%d\n", GetLastError());
  }
  return std::move(cwd);
}

string GetCwd() {
  unique_ptr<WCHAR[]> cwd(GetCwdW());
  return string(
      WstringToCstring(cwd.get() + (HasUncPrefix(cwd.get()) ? 4 : 0)).get());
}

bool ChangeDirectory(const string& path) {
  wstring wpath;
  if (!AsWindowsPathWithUncPrefix(path, &wpath)) {
    return false;
  }
  if (!::SetCurrentDirectoryW(wpath.c_str())) {
    PrintError(
        "ChangeDirectory(%s): SetCurrentDirectoryW(%S), failed, err=%d\n",
        path.c_str(), wpath.c_str(), GetLastError());
    return false;
  }
  return true;
}

#ifdef COMPILER_MSVC
void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ForEachDirectoryEntry is not implemented on Windows");
}
#else   // not COMPILER_MSVC
#endif  // COMPILER_MSVC

string NormalizeWindowsPath(string path) {
  if (path.empty()) {
    return "";
  }
  if (path[0] == '/') {
    // This is an absolute MSYS path, error out.
    pdie(255, "NormalizeWindowsPath: expected a Windows path, path=(%s)",
         path.c_str());
  }
  if (path.size() >= 4 && HasUncPrefix(path.c_str())) {
    path = path.substr(4);
  }

  static const string dot(".");
  static const string dotdot("..");

  vector<string> segments;
  int segment_start = -1;
  // Find the path segments in `path` (separated by "/").
  for (int i = 0;; ++i) {
    if (!IsPathSeparator(path[i]) && path[i] != '\0') {
      // The current character does not end a segment, so start one unless it's
      // already started.
      if (segment_start < 0) {
        segment_start = i;
      }
    } else if (segment_start >= 0 && i > segment_start) {
      // The current character is "/" or "\0", so this ends a segment.
      // Add that to `segments` if there's anything to add; handle "." and "..".
      string segment(path, segment_start, i - segment_start);
      segment_start = -1;
      if (segment == dotdot) {
        if (!segments.empty() &&
            !HasDriveSpecifierPrefix(segments[0].c_str())) {
          segments.pop_back();
        }
      } else if (segment != dot) {
        segments.push_back(segment);
      }
    }
    if (path[i] == '\0') {
      break;
    }
  }

  // Handle the case when `path` is just a drive specifier (or some degenerate
  // form of it, e.g. "c:\..").
  if (segments.size() == 1 && segments[0].size() == 2 &&
      HasDriveSpecifierPrefix(segments[0].c_str())) {
    return segments[0] + '\\';
  }

  // Join all segments.
  bool first = true;
  std::ostringstream result;
  for (const auto& s : segments) {
    if (!first) {
      result << '\\';
    }
    first = false;
    result << s;
  }
  return result.str();
}

}  // namespace blaze_util
