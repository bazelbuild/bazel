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
#include <windows.h>

#include <memory>  // unique_ptr
#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;
using std::wstring;

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

// Checks if the path is absolute and/or is a root path.
//
// If `must_be_root` is true, then in addition to being absolute, the path must
// also be just the root part, no other components, e.g. "c:\" is both absolute
// and root, but "c:\foo" is just absolute.
static bool IsRootOrAbsolute(const string& path, bool must_be_root) {
  // An absolute path is one that starts with "/", "\", "c:/", "c:\",
  // "\\?\c:\", or "\??\c:\".
  //
  // It is unclear whether the UNC prefix is just "\\?\" or is "\??\" also
  // valid (in some cases it seems to be, though MSDN doesn't mention it).
  return
      // path is (or starts with) "/" or "\"
      ((must_be_root ? path.size() == 1 : !path.empty()) &&
       (path[0] == '/' || path[0] == '\\')) ||
      // path is (or starts with) "c:/" or "c:\" or similar
      ((must_be_root ? path.size() == 3 : path.size() >= 3) &&
       isalpha(path[0]) && path[1] == ':' &&
       (path[2] == '/' || path[2] == '\\')) ||
      // path is (or starts with) "\\?\c:\" or "\??\c:\" or similar
      ((must_be_root ? path.size() == 7 : path.size() >= 7) &&
       path[0] == '\\' && (path[1] == '\\' || path[1] == '?') &&
       path[2] == '?' && path[3] == '\\' && isalpha(path[4]) &&
       path[5] == ':' && path[6] == '\\');
}

pair<string, string> SplitPath(const string& path) {
  if (path.empty()) {
    return std::make_pair("", "");
  }

  size_t pos = path.size() - 1;
  for (auto it = path.crbegin(); it != path.crend(); ++it, --pos) {
    if (*it == '/' || *it == '\\') {
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

bool AsWindowsPath(const string& path, wstring* result) {
  if (path.empty()) {
    result->clear();
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

  unique_ptr<WCHAR[]> mutable_wpath(CstringToWstring(mutable_path.c_str()));
  WCHAR* p = mutable_wpath.get();
  // Replace forward slashes with backslashes.
  while (*p != L'\0') {
    if (*p == L'/') {
      *p = L'\\';
    }
    ++p;
  }
  result->assign(mutable_wpath.get());
  return true;
}

bool ReadFile(const string& filename, string* content, int max_size) {
  wstring wfilename;
  if (!AsWindowsPath(filename, &wfilename)) {
    // Failed to convert the path because it was an absolute MSYS path but we
    // could not retrieve the BAZEL_SH envvar.
    return false;
  }

  if (wfilename.size() > MAX_PATH) {
    // CreateFileW requires that paths longer than MAX_PATH be prefixed with
    // "\\?\", so add that here.
    // TODO(laszlocsomor): add a test for this code path.
    wfilename = wstring(L"\\\\?\\") + wfilename;
  }

  HANDLE handle = CreateFileW(
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

#ifdef COMPILER_MSVC
bool WriteFile(const void* data, size_t size, const string& filename) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::WriteFile is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool UnlinkPath(const string& file_path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::UnlinkPath is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

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
  wstring wpath;
  if (!AsWindowsPath(NormalizePath(path), &wpath)) {
    PrintError("could not convert path to widechar, path=(%s), err=%d\n",
               path.c_str(), GetLastError());
    return false;
  }
  if (!IsAbsolute(path)) {
    DWORD len = ::GetCurrentDirectoryW(0, nullptr);
    unique_ptr<WCHAR[]> cwd(new WCHAR[len]);
    if (!GetCurrentDirectoryW(len, cwd.get())) {
      PrintError("could not make the path absolute, path=(%s), err=%d\n",
                 path.c_str(), GetLastError());
      return false;
    }
    wpath = wstring(cwd.get()) + L"\\" + wpath;
  }
  wpath = wstring(L"\\\\?\\") + wpath;
  return JunctionResolver().Resolve(wpath.c_str(), nullptr);
}

#ifdef COMPILER_MSVC
bool CanAccess(const string& path, bool read, bool write, bool exec) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::CanAccess is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool IsDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::IsDirectory is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

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

#ifdef COMPILER_MSVC
bool MakeDirectories(const string& path, unsigned int mode) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze::MakeDirectories is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
string GetCwd() {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetCwd is not implemented on Windows");
  return "";
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
bool ChangeDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ChangeDirectory is not implemented on Windows");
  return false;
}
#else  // not COMPILER_MSVC
#endif  // COMPILER_MSVC

#ifdef COMPILER_MSVC
void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ForEachDirectoryEntry is not implemented on Windows");
}
#else   // not COMPILER_MSVC
#endif  // COMPILER_MSVC

}  // namespace blaze_util
