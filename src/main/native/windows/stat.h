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

#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_STAT_H_
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_STAT_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <cstdint>
#include <string>

namespace bazel {
namespace windows {

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
struct StatResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kDoesNotExist = 2,
    kAccessDenied = 3,
  };
};

// Keep in sync with WindowsFileOperations#STAT_RESULT_LENGTH.
constexpr int kStatResultLength = 4;

// Times are in FILETIME units (100ns since 1601-01-01 UTC).
struct FileMetadata {
  DWORD attributes;
  int64_t size;
  LARGE_INTEGER last_write_time;
  LARGE_INTEGER change_time;
};

// Retrieves the attributes, size, last write time and change time of `path`.
//
// `path` should be an absolute, normalized, Windows-style path, with "\\?\"
// prefix if it's longer than MAX_PATH.
//
// Uses GetFileInformationByName (Windows 11 build 26100+) when it can, and a
// file handle otherwise. When the function returns StatResult::kError, `error`
// receives an error message.
int Stat(const std::wstring& path, bool follow_reparse_points,
         FileMetadata& result, std::wstring& error);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_STAT_H_
