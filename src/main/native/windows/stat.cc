// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include "src/main/native/windows/stat.h"

#include <windows.h>

#include <string>

#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

namespace {

// From the 10.0.26100.0 SDK. Declared here because older SDKs don't have them.
struct FILE_STAT_BASIC_INFORMATION {
  LARGE_INTEGER FileId;
  LARGE_INTEGER CreationTime;
  LARGE_INTEGER LastAccessTime;
  LARGE_INTEGER LastWriteTime;
  LARGE_INTEGER ChangeTime;
  LARGE_INTEGER AllocationSize;
  LARGE_INTEGER EndOfFile;
  DWORD FileAttributes;
  DWORD ReparseTag;
  DWORD NumberOfLinks;
  DWORD DeviceType;
  DWORD DeviceCharacteristics;
  DWORD Reserved;
  LARGE_INTEGER VolumeSerialNumber;
  FILE_ID_128 FileId128;
};

static_assert(sizeof(FILE_STAT_BASIC_INFORMATION) == 104,
              "FILE_STAT_BASIC_INFORMATION must match the SDK's layout");

enum FILE_INFO_BY_NAME_CLASS { FileStatBasicByNameInfo = 3 };

// Looked up at runtime: a static import would keep Bazel from starting on
// Windows 10.
using GetFileInformationByNameFn = BOOL(WINAPI*)(PCWSTR,
                                                 FILE_INFO_BY_NAME_CLASS, PVOID,
                                                 ULONG);

// Returns null below Windows 11 build 26100.
GetFileInformationByNameFn GetFileInformationByNameOrNull() {
  static const GetFileInformationByNameFn fn =
      reinterpret_cast<GetFileInformationByNameFn>(::GetProcAddress(
          ::GetModuleHandleW(L"kernel32.dll"), "GetFileInformationByName"));
  return fn;
}

// Fallback for when GetFileInformationByName is unavailable or can't answer.
int StatByHandle(const WCHAR* path, bool follow_reparse_points,
                 FileMetadata& result, std::wstring& error) {
  DWORD flags = FILE_FLAG_BACKUP_SEMANTICS;
  if (!follow_reparse_points) {
    flags |= FILE_FLAG_OPEN_REPARSE_POINT;
  }
  AutoHandle handle(::CreateFileW(
      path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, flags, nullptr));
  if (!handle.IsValid()) {
    DWORD err = ::GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      return StatResult::kDoesNotExist;
    }
    // A reparse point that is not a link, e.g. an App Execution Alias or an
    // AF_UNIX socket, can't be followed: treat it like a dangling link.
    if (err == ERROR_CANT_ACCESS_FILE && follow_reparse_points) {
      return StatResult::kDoesNotExist;
    }
    if (err == ERROR_ACCESS_DENIED) {
      return StatResult::kAccessDenied;
    }
    error =
        MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateFileW", path, err);
    return StatResult::kError;
  }

  FILE_BASIC_INFO basic;
  FILE_STANDARD_INFO standard;
  if (!::GetFileInformationByHandleEx(handle, FileBasicInfo, &basic,
                                      sizeof(basic)) ||
      !::GetFileInformationByHandleEx(handle, FileStandardInfo, &standard,
                                      sizeof(standard))) {
    DWORD err = ::GetLastError();
    error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                             L"GetFileInformationByHandleEx", path, err);
    return StatResult::kError;
  }

  result = {
      // Following resolves only links. Any other reparse point is handled by a
      // filter driver, which opens the file itself and may leave this bit set.
      follow_reparse_points
          ? basic.FileAttributes & ~FILE_ATTRIBUTE_REPARSE_POINT
          : basic.FileAttributes,
      standard.EndOfFile.QuadPart, basic.LastWriteTime, basic.ChangeTime};
  return StatResult::kSuccess;
}

}  // namespace

int Stat(const std::wstring& path, bool follow_reparse_points,
         FileMetadata& result, std::wstring& error) {
  if (!IsAbsoluteNormalizedWindowsPath(path)) {
    error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"Stat", path,
                             L"expected an absolute Windows path");
    return StatResult::kError;
  }

  GetFileInformationByNameFn stat_by_name =
      GetFileInformationByNameOrNull();
  if (stat_by_name == nullptr) {
    return StatByHandle(path.c_str(), follow_reparse_points, result, error);
  }

  FILE_STAT_BASIC_INFORMATION info;
  if (!stat_by_name(path.c_str(), FileStatBasicByNameInfo, &info,
                    sizeof(info))) {
    // Other errors (unsupported filesystem, unmounted volume) go to the
    // fallback rather than being reported as a missing file.
    DWORD err = ::GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      return StatResult::kDoesNotExist;
    }
    return StatByHandle(path.c_str(), follow_reparse_points, result, error);
  }

  // GetFileInformationByName doesn't follow reparse points.
  if (follow_reparse_points &&
      (info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
    return StatByHandle(path.c_str(), follow_reparse_points, result, error);
  }

  result = {info.FileAttributes, info.EndOfFile.QuadPart, info.LastWriteTime,
            info.ChangeTime};
  return StatResult::kSuccess;
}

}  // namespace windows
}  // namespace bazel
