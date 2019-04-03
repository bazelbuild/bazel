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

#include <windows.h>
#include <WinIoCtl.h>

#include <stdint.h>  // uint8_t

#include <memory>
#include <sstream>
#include <string>

#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

using std::unique_ptr;
using std::wstring;

wstring AddUncPrefixMaybe(const wstring& path) {
  return path.empty() || IsDevNull(path.c_str()) || HasUncPrefix(path.c_str())
             ? path
             : (wstring(L"\\\\?\\") + path);
}

wstring RemoveUncPrefixMaybe(const wstring& path) {
  return bazel::windows::HasUncPrefix(path.c_str()) ? path.substr(4) : path;
}

bool HasDriveSpecifierPrefix(const wstring& p) {
  if (HasUncPrefix(p.c_str())) {
    return p.size() >= 7 && iswalpha(p[4]) && p[5] == ':' && p[6] == '\\';
  } else {
    return p.size() >= 3 && iswalpha(p[0]) && p[1] == ':' && p[2] == '\\';
  }
}

bool IsAbsoluteNormalizedWindowsPath(const wstring& p) {
  if (p.empty()) {
    return false;
  }
  if (IsDevNull(p.c_str())) {
    return true;
  }
  if (p.find_first_of('/') != wstring::npos) {
    return false;
  }

  return HasDriveSpecifierPrefix(p) && p.find(L".\\") != 0 &&
         p.find(L"\\.\\") == wstring::npos && p.find(L"\\.") != p.size() - 2 &&
         p.find(L"..\\") != 0 && p.find(L"\\..\\") == wstring::npos &&
         p.find(L"\\..") != p.size() - 3;
}

static wstring uint32asHexString(uint32_t value) {
  WCHAR attr_str[8];
  for (int i = 0; i < 8; ++i) {
    attr_str[7 - i] = L"0123456789abcdef"[value & 0xF];
    value >>= 4;
  }
  return wstring(attr_str, 8);
}

int IsSymlinkOrJunction(const WCHAR* path, bool* result, wstring* error) {
  if (!IsAbsoluteNormalizedWindowsPath(path)) {
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"IsSymlinkOrJunction", path,
                                L"expected an absolute Windows path");
    }
    return IsSymlinkOrJunctionResult::kError;
  }

  DWORD attrs = ::GetFileAttributesW(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    DWORD err = GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      return IsSymlinkOrJunctionResult::kDoesNotExist;
    }

    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"IsSymlinkOrJunction", path, err);
    }
    return IsSymlinkOrJunctionResult::kError;
  } else {
    *result = (attrs & FILE_ATTRIBUTE_REPARSE_POINT);
    return IsSymlinkOrJunctionResult::kSuccess;
  }
}

wstring GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  if (!IsAbsoluteNormalizedWindowsPath(path)) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"GetLongPath", path,
                            L"expected an absolute Windows path");
  }

  std::wstring wpath(AddUncPrefixMaybe(path));
  DWORD size = ::GetLongPathNameW(wpath.c_str(), NULL, 0);
  if (size == 0) {
    DWORD err_code = GetLastError();
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"GetLongPathNameW", path,
                            err_code);
  }
  result->reset(new WCHAR[size]);
  ::GetLongPathNameW(wpath.c_str(), result->get(), size);
  return L"";
}

#pragma pack(push, 4)
// See https://msdn.microsoft.com/en-us/windows/desktop/ff552012
typedef struct _REPARSE_DATA_BUFFER {
  ULONG ReparseTag;
  USHORT ReparseDataLength;
  USHORT Reserved;
  union {
    struct {
      USHORT SubstituteNameOffset;
      USHORT SubstituteNameLength;
      USHORT PrintNameOffset;
      USHORT PrintNameLength;
      ULONG Flags;
      WCHAR PathBuffer[1];
    } SymbolicLinkReparseBuffer;
    struct {
      USHORT SubstituteNameOffset;
      USHORT SubstituteNameLength;
      USHORT PrintNameOffset;
      USHORT PrintNameLength;
      WCHAR PathBuffer[1];
    } MountPointReparseBuffer;
    struct {
      UCHAR DataBuffer[1];
    } GenericReparseBuffer;
  } DUMMYUNIONNAME;
} REPARSE_DATA_BUFFER, *PREPARSE_DATA_BUFFER;
#pragma pack(pop)

int CreateJunction(const wstring& junction_name, const wstring& junction_target,
                   wstring* error) {
  if (!IsAbsoluteNormalizedWindowsPath(junction_name)) {
    if (error) {
      *error = MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"CreateJunction", junction_name,
          L"expected an absolute Windows path for junction_name");
    }
    return CreateJunctionResult::kError;
  }
  if (!IsAbsoluteNormalizedWindowsPath(junction_target)) {
    if (error) {
      *error = MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"CreateJunction", junction_target,
          L"expected an absolute Windows path for junction_target");
    }
    return CreateJunctionResult::kError;
  }

  const WCHAR* target = HasUncPrefix(junction_target.c_str())
                            ? junction_target.c_str() + 4
                            : junction_target.c_str();
  const size_t target_size = HasUncPrefix(junction_target.c_str())
                                 ? junction_target.size() - 4
                                 : junction_target.size();
  // The entire REPARSE_DATA_BUFFER cannot be larger than
  // MAXIMUM_REPARSE_DATA_BUFFER_SIZE bytes.
  //
  // The structure's layout is:
  //   [8 bytes] : ReparseTag, ReparseDataLength, Reserved
  //   [8 bytes] : MountPointReparseBuffer members before PathBuffer
  //   ---- start of MountPointReparseBuffer.PathBuffer ----
  //   [4 WCHARs]             : "\??\" prefix
  //   [target.size() WCHARs] : junction target name
  //   [1 WCHAR]              : null-terminator
  //   [target.size() WCHARs] : junction target displayed name
  //   [1 WCHAR]              : null-terminator
  // The sum of these must not exceed MAXIMUM_REPARSE_DATA_BUFFER_SIZE.
  // We can rearrange this to get the limit for target.size().
  static const size_t kMaxJunctionTargetLen =
      ((MAXIMUM_REPARSE_DATA_BUFFER_SIZE -
        offsetof(REPARSE_DATA_BUFFER, MountPointReparseBuffer.PathBuffer)) /
           sizeof(WCHAR) -
       /* one "\??\" prefix */ 4 -
       /* two null terminators */ 2) /
      /* two copies of the string are stored */ 2;
  if (target_size > kMaxJunctionTargetLen) {
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateJunction",
                                target, L"target path is too long");
    }
    return CreateJunctionResult::kTargetNameTooLong;
  }
  const wstring name = HasUncPrefix(junction_name.c_str())
                           ? junction_name
                           : (wstring(L"\\\\?\\") + junction_name);

  // Junctions are directories, so create a directory.
  // If CreateDirectoryW succeeds, we'll try to set the junction's target.
  // If CreateDirectoryW fails, we don't care about the exact reason -- could be
  // that the directory already exists, or we have no access to create a
  // directory, or the path was invalid to begin with. Either way set `create`
  // to false, meaning we'll just attempt to open the path for metadata-reading
  // and check if it's a junction pointing to the desired target.
  bool create = CreateDirectoryW(name.c_str(), NULL) != 0;

  AutoHandle handle;
  if (create) {
    handle = CreateFileW(
        name.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, NULL);
  }

  if (!handle.IsValid()) {
    // We can't open the directory for writing: either we didn't even try to
    // (`create` was false), or the path disappeared, or it turned into a file,
    // or another process holds it open without write-sharing.
    // Either way, don't try to create the junction, just try opening it without
    // any read or write access (we can still read its metadata) and maximum
    // sharing, and check its target.
    create = false;
    handle = CreateFileW(
        name.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (!handle.IsValid()) {
      // We can't open the directory at all: either it disappeared, or it turned
      // into a file, or the path is invalid, or another process holds it open
      // without any sharing. Give up.
      DWORD err = GetLastError();
      if (err == ERROR_SHARING_VIOLATION) {
        // The junction is held open by another process.
        return CreateJunctionResult::kAccessDenied;
      } else if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
        // Meanwhile the directory disappeared or one of its parent directories
        // disappeared.
        return CreateJunctionResult::kDisappeared;
      }

      // The path seems to exist yet we cannot open it for metadata-reading.
      // Report as much information as we have, then give up.
      if (error) {
        *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateFileW",
                                  name, err);
      }
      return CreateJunctionResult::kError;
    }
  }

  // We have an open handle to the file! It may still be other than a junction,
  // so check its attributes.
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(handle, &info)) {
    DWORD err = GetLastError();
    // Some unknown error occurred.
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"GetFileInformationByHandle", name, err);
    }
    return CreateJunctionResult::kError;
  }

  if (info.dwFileAttributes == INVALID_FILE_ATTRIBUTES) {
    DWORD err = GetLastError();
    // Some unknown error occurred.
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"GetFileInformationByHandle", name, err);
    }
    return CreateJunctionResult::kError;
  }

  if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    // The path already exists and it's a junction. Do not overwrite, just check
    // its target.
    create = false;
  }

  if (create) {
    if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
      // Even though we managed to create the directory and it didn't exist
      // before, another process changed it in the meantime so it's no longer a
      // directory.
      create = false;
      if (!(info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
        // The path is no longer a directory, and it's not a junction either.
        // Though this is a case for kAlreadyExistsButNotJunction, let's instead
        // print the attributes and return kError, to give more information to
        // the user.
        if (error) {
          *error = MakeErrorMessage(
              WSTR(__FILE__), __LINE__, L"GetFileInformationByHandle", name,
              wstring(L"attrs=0x") + uint32asHexString(info.dwFileAttributes));
        }
        return CreateJunctionResult::kError;
      }
    }
  }

  if (!create) {
    // The path already exists. Check if it's a junction.
    if (!(info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
      return CreateJunctionResult::kAlreadyExistsButNotJunction;
    }
  }

  uint8_t reparse_buffer_bytes[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
  PREPARSE_DATA_BUFFER reparse_buffer =
      reinterpret_cast<PREPARSE_DATA_BUFFER>(reparse_buffer_bytes);
  if (create) {
    // The junction doesn't exist yet, and we have an open handle to the
    // candidate directory with write access and no sharing. Proceed to turn the
    // directory into a junction.

    memset(reparse_buffer_bytes, 0, MAXIMUM_REPARSE_DATA_BUFFER_SIZE);

    reparse_buffer->MountPointReparseBuffer.SubstituteNameOffset = 0;
    reparse_buffer->MountPointReparseBuffer.SubstituteNameLength =
        (4 + target_size) * sizeof(WCHAR);
    reparse_buffer->MountPointReparseBuffer.PrintNameOffset =
        reparse_buffer->MountPointReparseBuffer.SubstituteNameLength +
        /* null-terminator */ sizeof(WCHAR);
    reparse_buffer->MountPointReparseBuffer.PrintNameLength =
        target_size * sizeof(WCHAR);

    reparse_buffer->ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
    reparse_buffer->ReparseDataLength =
        4 * sizeof(USHORT) +
        reparse_buffer->MountPointReparseBuffer.SubstituteNameLength +
        reparse_buffer->MountPointReparseBuffer.PrintNameLength +
        /* 2 null-terminators */ (2 * sizeof(WCHAR));
    reparse_buffer->Reserved = 0;

    // "\??\" is meaningful to the kernel, it's a synomym for the "\DosDevices\"
    // object path. (NOT to be confused with "\\?\" which is meaningful for the
    // Win32 API.) We need to use this prefix to tell the kernel where the
    // reparse point is pointing to.
    memcpy((uint8_t*)reparse_buffer->MountPointReparseBuffer.PathBuffer +
               reparse_buffer->MountPointReparseBuffer.SubstituteNameOffset,
           L"\\??\\", 4 * sizeof(WCHAR));
    memcpy((uint8_t*)reparse_buffer->MountPointReparseBuffer.PathBuffer +
               reparse_buffer->MountPointReparseBuffer.SubstituteNameOffset +
               4 * sizeof(WCHAR),
           target,
           reparse_buffer->MountPointReparseBuffer.SubstituteNameLength -
               4 * sizeof(WCHAR));

    // In addition to their target, junctions also have another string which is
    // a user-visible name of where the junction points, as listed by "dir".
    // This can be any string and won't affect the usability of the junction.
    // MKLINK uses the target path without the "\??\" prefix as the display
    // name, so let's do that here too. This is also in line with how UNIX
    // behaves. Using a dummy or fake display name would be misleading, it would
    // make the output of `dir` look like:
    //   2017-01-18  01:37 PM    <JUNCTION>     juncname [dummy string]
    memcpy((uint8_t*)reparse_buffer->MountPointReparseBuffer.PathBuffer +
               reparse_buffer->MountPointReparseBuffer.PrintNameOffset,
           target, reparse_buffer->MountPointReparseBuffer.PrintNameLength);

    DWORD bytes_returned;
    if (!::DeviceIoControl(
            handle, FSCTL_SET_REPARSE_POINT, reparse_buffer,
            reparse_buffer->ReparseDataLength +
                offsetof(REPARSE_DATA_BUFFER, GenericReparseBuffer.DataBuffer),
            NULL, 0, &bytes_returned, NULL)) {
      DWORD err = GetLastError();
      if (err == ERROR_DIR_NOT_EMPTY) {
        return CreateJunctionResult::kAlreadyExistsButNotJunction;
      }
      // Some unknown error occurred.
      if (error) {
        *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeviceIoControl",
                                  name, err);
      }
      return CreateJunctionResult::kError;
    }
  } else {
    // The junction already exists. Check if it points to the right target.

    DWORD bytes_returned;
    if (!::DeviceIoControl(handle, FSCTL_GET_REPARSE_POINT, NULL, 0,
                           reparse_buffer, MAXIMUM_REPARSE_DATA_BUFFER_SIZE,
                           &bytes_returned, NULL)) {
      DWORD err = GetLastError();
      // Some unknown error occurred.
      if (error) {
        *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeviceIoControl",
                                  name, err);
      }
      return CreateJunctionResult::kError;
    }

    WCHAR* actual_target =
        reparse_buffer->MountPointReparseBuffer.PathBuffer +
        reparse_buffer->MountPointReparseBuffer.SubstituteNameOffset +
        /* "\??\" prefix */ 4;
    if (reparse_buffer->MountPointReparseBuffer.SubstituteNameLength !=
            (/* "\??\" prefix */ 4 + target_size) * sizeof(WCHAR) ||
        _wcsnicmp(actual_target, target, target_size) != 0) {
      return CreateJunctionResult::kAlreadyExistsWithDifferentTarget;
    }
  }

  return CreateJunctionResult::kSuccess;
}

int ReadSymlinkOrJunction(const wstring& path, wstring* result,
                          wstring* error) {
  if (!IsAbsoluteNormalizedWindowsPath(path)) {
    if (error) {
      *error = MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"ReadSymlinkOrJunction", path,
          L"expected an absolute Windows path for 'path'");
    }
    return ReadSymlinkOrJunctionResult::kError;
  }

  AutoHandle handle(CreateFileW(
      AddUncPrefixMaybe(path).c_str(), 0,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
      OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
      NULL));
  if (!handle.IsValid()) {
    DWORD err = GetLastError();
    if (err == ERROR_SHARING_VIOLATION) {
      // The path is held open by another process.
      return ReadSymlinkOrJunctionResult::kAccessDenied;
    } else if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      // Path or a parent directory does not exist.
      return ReadSymlinkOrJunctionResult::kDoesNotExist;
    }

    // The path seems to exist yet we cannot open it for metadata-reading.
    // Report as much information as we have, then give up.
    if (error) {
      *error =
          MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateFileW", path, err);
    }
    return ReadSymlinkOrJunctionResult::kError;
  }

  uint8_t raw_buf[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
  PREPARSE_DATA_BUFFER buf = reinterpret_cast<PREPARSE_DATA_BUFFER>(raw_buf);
  DWORD bytes_returned;
  if (!::DeviceIoControl(handle, FSCTL_GET_REPARSE_POINT, NULL, 0, buf,
                         MAXIMUM_REPARSE_DATA_BUFFER_SIZE, &bytes_returned,
                         NULL)) {
    DWORD err = GetLastError();
    if (err == ERROR_NOT_A_REPARSE_POINT) {
      return ReadSymlinkOrJunctionResult::kNotALink;
    }

    // Some unknown error occurred.
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeviceIoControl",
                                path, err);
    }
    return ReadSymlinkOrJunctionResult::kError;
  }

  switch (buf->ReparseTag) {
    case IO_REPARSE_TAG_SYMLINK: {
      wchar_t* p =
          (wchar_t*)(((uint8_t*)buf->SymbolicLinkReparseBuffer.PathBuffer) +
                     buf->SymbolicLinkReparseBuffer.SubstituteNameOffset);
      *result = wstring(p, buf->SymbolicLinkReparseBuffer.SubstituteNameLength /
                               sizeof(WCHAR));
      return ReadSymlinkOrJunctionResult::kSuccess;
    }
    case IO_REPARSE_TAG_MOUNT_POINT: {
      wchar_t* p =
          (wchar_t*)(((uint8_t*)buf->MountPointReparseBuffer.PathBuffer) +
                     buf->MountPointReparseBuffer.SubstituteNameOffset);
      *result = wstring(
          p, buf->MountPointReparseBuffer.SubstituteNameLength / sizeof(WCHAR));
      return ReadSymlinkOrJunctionResult::kSuccess;
    }
    default:
      return ReadSymlinkOrJunctionResult::kUnknownLinkType;
  }
}

struct DirectoryStatus {
  enum {
    kDoesNotExist = 0,
    kDirectoryEmpty = 1,
    kDirectoryNotEmpty = 2,
    kChildMarkedForDeletionExists = 3,
  };
};

// Check whether the directory and its child elements truly exist, or are marked
// for deletion. The result could be:
// 1. The give path doesn't exist
// 2. The directory is empty
// 3. The directory contains valid files or dirs, so not empty
// 4. The directory contains only files or dirs marked for deletion.
int CheckDirectoryStatus(const wstring& path) {
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");
  bool found_valid_file = false;
  bool found_child_marked_for_deletion = false;
  WIN32_FIND_DATAW metadata;
  HANDLE handle = ::FindFirstFileW((path + L"\\*").c_str(), &metadata);
  if (handle == INVALID_HANDLE_VALUE) {
    return DirectoryStatus::kDoesNotExist;
  }
  do {
    if (kDot != metadata.cFileName && kDotDot != metadata.cFileName) {
      std::wstring child = path + L"\\" + metadata.cFileName;
      DWORD attributes = GetFileAttributesW(child.c_str());
      if (attributes != INVALID_FILE_ATTRIBUTES) {
        // If there is a valid file under the directory,
        // then the directory is truely not empty.
        // We should just return kDirectoryNotEmpty.
        found_valid_file = true;
        break;
      } else {
        DWORD error_code = GetLastError();
        // If the file or directory is in deleting process,
        // GetFileAttributesW returns ERROR_ACCESS_DENIED,
        // If it's already deleted at the time we check,
        // GetFileAttributesW returns ERROR_FILE_NOT_FOUND.
        // If GetFileAttributesW fails with other reason, we consider there is a
        // valid file that we cannot open, thus return kDirectoryNotEmpty
        if (error_code != ERROR_ACCESS_DENIED &&
            error_code != ERROR_FILE_NOT_FOUND) {
          found_valid_file = true;
          break;
        } else if (error_code == ERROR_ACCESS_DENIED) {
          found_child_marked_for_deletion = true;
        }
      }
    }
  } while (::FindNextFileW(handle, &metadata));
  ::FindClose(handle);
  if (found_valid_file) {
    return DirectoryStatus::kDirectoryNotEmpty;
  }
  if (found_child_marked_for_deletion) {
    return DirectoryStatus::kChildMarkedForDeletionExists;
  }
  return DirectoryStatus::kDirectoryEmpty;
}

int DeletePath(const wstring& path, wstring* error) {
  if (!IsAbsoluteNormalizedWindowsPath(path)) {
    if (error) {
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeletePath", path,
                                L"expected an absolute Windows path");
    }
    return DeletePathResult::kError;
  }

  const std::wstring winpath(AddUncPrefixMaybe(path));
  const wchar_t* wpath = winpath.c_str();

  if (!DeleteFileW(wpath)) {
    DWORD err = GetLastError();
    if (err == ERROR_SHARING_VIOLATION) {
      // The file or directory is in use by some process or we have no
      // permission to delete it.
      return DeletePathResult::kAccessDenied;
    } else if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      // The file or directory does not exist, or a parent directory does not
      // exist, or a parent directory is actually a file.
      return DeletePathResult::kDoesNotExist;
    } else if (err != ERROR_ACCESS_DENIED) {
      // Some unknown error occurred.
      if (error) {
        *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeleteFileW",
                                  path, err);
      }
      return DeletePathResult::kError;
    }

    // DeleteFileW failed with access denied, because the file is read-only or
    // it is a directory.
    DWORD attr = GetFileAttributesW(wpath);
    if (attr == INVALID_FILE_ATTRIBUTES) {
      err = GetLastError();
      if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
        // The file disappeared, or one of its parent directories disappeared,
        // or one of its parent directories is no longer a directory.
        return DeletePathResult::kDoesNotExist;
      } else {
        // Some unknown error occurred.
        if (error) {
          *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                    L"GetFileAttributesW", path, err);
        }
        return DeletePathResult::kError;
      }
    }

    // DeleteFileW failed with access denied, but the path exists.
    if (attr & FILE_ATTRIBUTE_DIRECTORY) {
      // It's a directory or a junction.

      // Sometimes a deleted directory lingers in its parent dir
      // after the deleting handle has already been closed.
      // In this case we check the content of the parent directory,
      // if we don't find any valid file, we try to delete it again after 5 ms.
      // But we don't want to hang infinitely because another application
      // can hold the handle for a long time. So try at most 20 times,
      // which means a process time of 100-120ms.
      // Inspired by
      // https://github.com/Alexpux/Cygwin/commit/28fa2a72f810670a0562ea061461552840f5eb70
      // Useful link: https://stackoverflow.com/questions/31606978
      int count = 20;
      while (count > 0 && !RemoveDirectoryW(wpath)) {
        // Failed to delete the directory.
        err = GetLastError();
        if (err == ERROR_SHARING_VIOLATION || err == ERROR_ACCESS_DENIED) {
          // The junction or directory is in use by another process, or we have
          // no permission to delete it.
          return DeletePathResult::kAccessDenied;
        } else if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
          // The directory or one of its directories disappeared or is no longer
          // a directory.
          return DeletePathResult::kDoesNotExist;
        } else if (err == ERROR_DIR_NOT_EMPTY) {
          // We got ERROR_DIR_NOT_EMPTY error, but maybe the child files and
          // dirs are already marked for deletion, let's check the status of the
          // child elements to see if we should retry the delete operation.
          switch (CheckDirectoryStatus(winpath)) {
            case DirectoryStatus::kDirectoryNotEmpty:
              // The directory is truely not empty.
              return DeletePathResult::kDirectoryNotEmpty;
            case DirectoryStatus::kDirectoryEmpty:
              // If we didn't find any pending delete child files or dirs, it
              // means at the time we check, the child elements marked for
              // deletion are gone, the directory is now empty, we can then try
              // to delete the directory again without waiting.
              continue;
            case DirectoryStatus::kChildMarkedForDeletionExists:
              // If the directory only contains child elements marked for
              // deletion, wait the system for 5 ms to clean them up and try to
              // delete the directory again.
              Sleep(5L);
              continue;
            case DirectoryStatus::kDoesNotExist:
              // This case should never happen, because ERROR_DIR_NOT_EMPTY
              // means the directory exists. But if it does happen, return an
              // error message.
              if (error) {
                *error =
                    MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                     L"RemoveDirectoryW", path, GetLastError());
              }
              return DeletePathResult::kError;
          }
        }

        // Some unknown error occurred.
        if (error) {
          *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                    L"RemoveDirectoryW", path, err);
        }
        return DeletePathResult::kError;
      }

      if (count == 0) {
        // After trying 20 times, the "deleted" sub-directories or files still
        // won't go away, so just return kDirectoryNotEmpty error.
        return DeletePathResult::kDirectoryNotEmpty;
      }
    } else if (attr & FILE_ATTRIBUTE_READONLY) {
      // It's a file and it's probably read-only.
      // Make it writable then try deleting it again.
      attr &= ~FILE_ATTRIBUTE_READONLY;
      if (!SetFileAttributesW(wpath, attr)) {
        err = GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
          // The file disappeared, or one of its parent directories disappeared,
          // or one of its parent directories is no longer a directory.
          return DeletePathResult::kDoesNotExist;
        }
        // Some unknown error occurred.
        if (error) {
          *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                    L"SetFileAttributesW", path, err);
        }
        return DeletePathResult::kError;
      }

      if (!DeleteFileW(wpath)) {
        // Failed to delete the file again.
        err = GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
          // The file disappeared, or one of its parent directories disappeared,
          // or one of its parent directories is no longer a directory.
          return DeletePathResult::kDoesNotExist;
        }

        // Some unknown error occurred.
        if (error) {
          *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeleteFileW",
                                    path, err);
        }
        return DeletePathResult::kError;
      }
    } else {
      if (error) {
        *error = MakeErrorMessage(
            WSTR(__FILE__), __LINE__,
            (std::wstring(L"Unknown error, winpath=[") + winpath + L"]")
                .c_str(),
            path, err);
      }
      return DeletePathResult::kError;
    }
  }
  return DeletePathResult::kSuccess;
}

}  // namespace windows
}  // namespace bazel
