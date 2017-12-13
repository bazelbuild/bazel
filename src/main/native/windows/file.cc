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

#include <stdint.h>  // uint8_t
#include <windows.h>

#include <memory>
#include <sstream>
#include <string>

#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

using std::unique_ptr;
using std::wstring;

int IsJunctionOrDirectorySymlink(const WCHAR* path) {
  DWORD attrs = ::GetFileAttributesW(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return IS_JUNCTION_ERROR;
  } else {
    if ((attrs & FILE_ATTRIBUTE_DIRECTORY) &&
        (attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
      return IS_JUNCTION_YES;
    } else {
      return IS_JUNCTION_NO;
    }
  }
}

wstring GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  DWORD size = ::GetLongPathNameW(path, NULL, 0);
  if (size == 0) {
    DWORD err_code = GetLastError();
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"GetLongPathNameW", path,
                            err_code);
  }
  result->reset(new WCHAR[size]);
  ::GetLongPathNameW(path, result->get(), size);
  return L"";
}

HANDLE OpenDirectory(const WCHAR* path, bool read_write) {
  return ::CreateFileW(
      /* lpFileName */ path,
      /* dwDesiredAccess */
      read_write ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ,
      /* dwShareMode */ 0,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_FLAG_OPEN_REPARSE_POINT |
          FILE_FLAG_BACKUP_SEMANTICS,
      /* hTemplateFile */ NULL);
}

#pragma pack(push, 4)
typedef struct _JunctionDescription {
  typedef struct _Header {
    DWORD ReparseTag;
    WORD ReparseDataLength;
    WORD Reserved;
  } Header;

  typedef struct _WriteDesc {
    WORD SubstituteNameOffset;
    WORD SubstituteNameLength;
    WORD PrintNameOffset;
    WORD PrintNameLength;
  } WriteDesc;

  Header header;
  WriteDesc write;
  WCHAR PathBuffer[ANYSIZE_ARRAY];
} JunctionDescription;
#pragma pack(pop)

wstring CreateJunction(const wstring& junction_name,
                       const wstring& junction_target) {
  const wstring target = HasUncPrefix(junction_target.c_str())
                             ? junction_target.substr(4)
                             : junction_target;
  // The entire JunctionDescription cannot be larger than
  // MAXIMUM_REPARSE_DATA_BUFFER_SIZE bytes.
  //
  // The structure's layout is:
  //   [JunctionDescription::Header]
  //   [JunctionDescription::WriteDesc]
  //   ---- start of JunctionDescription::PathBuffer ----
  //   [4 WCHARs]             : "\??\" prefix
  //   [target.size() WCHARs] : junction target name
  //   [1 WCHAR]              : null-terminator
  //   [target.size() WCHARs] : junction target displayed name
  //   [1 WCHAR]              : null-terminator
  // The sum of these must not exceed MAXIMUM_REPARSE_DATA_BUFFER_SIZE.
  // We can rearrange this to get the limit for target.size().
  static const size_t kMaxJunctionTargetLen =
      ((MAXIMUM_REPARSE_DATA_BUFFER_SIZE - sizeof(JunctionDescription::Header) -
        sizeof(JunctionDescription::WriteDesc) -
        /* one "\??\" prefix */ sizeof(WCHAR) * 4 -
        /* two null terminators */ sizeof(WCHAR) * 2) /
       /* two copies of the string are stored */ 2) /
      sizeof(WCHAR);
  if (target.size() > kMaxJunctionTargetLen) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateJunction", target,
                            L"target is too long");
  }
  const wstring name = HasUncPrefix(junction_name.c_str())
                           ? junction_name
                           : (wstring(L"\\\\?\\") + junction_name);

  // Junctions are directories, so create one
  if (!::CreateDirectoryW(name.c_str(), NULL)) {
    DWORD err_code = GetLastError();
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateJunction", name,
                            err_code);
  }

  AutoHandle handle(OpenDirectory(name.c_str(), true));
  if (!handle.IsValid()) {
    DWORD err_code = GetLastError();
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"OpenDirectory", name,
                            err_code);
  }

  uint8_t reparse_buffer_bytes[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
  JunctionDescription* reparse_buffer =
      reinterpret_cast<JunctionDescription*>(reparse_buffer_bytes);
  memset(reparse_buffer_bytes, 0, MAXIMUM_REPARSE_DATA_BUFFER_SIZE);

  // "\??\" is meaningful to the kernel, it's a synomym for the "\DosDevices\"
  // object path. (NOT to be confused with "\\?\" which is meaningful for the
  // Win32 API.) We need to use this prefix to tell the kernel where the reparse
  // point is pointing to.
  memcpy(reparse_buffer->PathBuffer, L"\\??\\", 4 * sizeof(WCHAR));
  memcpy(reparse_buffer->PathBuffer + 4, target.c_str(),
         target.size() * sizeof(WCHAR));

  // In addition to their target, junctions also have another string which is a
  // user-visible name of where the junction points, as listed by "dir". This
  // can be any string and won't affect the usability of the junction.
  // MKLINK uses the target path without the "\??\" prefix as the display name,
  // so let's do that here too. This is also in line with how UNIX behaves.
  // Using a dummy or fake display name would be pure evil, it would make the
  // output of `dir` look like:
  //   2017-01-18  01:37 PM    <JUNCTION>     juncname [dummy string]
  memcpy(reparse_buffer->PathBuffer + 4 + target.size() + 1, target.c_str(),
         target.size() * sizeof(WCHAR));

  reparse_buffer->write.SubstituteNameOffset = 0;
  reparse_buffer->write.SubstituteNameLength =
      (4 + target.size()) * sizeof(WCHAR);
  reparse_buffer->write.PrintNameOffset =
      reparse_buffer->write.SubstituteNameLength +
      /* null-terminator */ sizeof(WCHAR);
  reparse_buffer->write.PrintNameLength = target.size() * sizeof(WCHAR);

  reparse_buffer->header.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
  reparse_buffer->header.ReparseDataLength =
      sizeof(JunctionDescription::WriteDesc) +
      reparse_buffer->write.SubstituteNameLength +
      reparse_buffer->write.PrintNameLength +
      /* 2 null-terminators */ (2 * sizeof(WCHAR));
  reparse_buffer->header.Reserved = 0;

  DWORD bytes_returned;
  if (!::DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, reparse_buffer,
                         reparse_buffer->header.ReparseDataLength +
                             sizeof(JunctionDescription::Header),
                         NULL, 0, &bytes_returned, NULL)) {
    DWORD err_code = GetLastError();
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"DeviceIoControl", L"",
                            err_code);
  }
  return L"";
}

}  // namespace windows
}  // namespace bazel
