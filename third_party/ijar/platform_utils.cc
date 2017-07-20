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

#include "third_party/ijar/platform_utils.h"

#include <limits.h>
#include <stdio.h>

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
#include <windows.h>
#else  // !(defined(COMPILER_MSVC) || defined(__CYGWIN__))
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

#include <string>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"

namespace devtools_ijar {

using std::string;

bool stat_file(const char* path, Stat* result) {
#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
  std::wstring wpath;
  if (!blaze_util::AsWindowsPathWithUncPrefix(path, &wpath)) {
    blaze_util::pdie(255, "stat_file: AsWindowsPathWithUncPrefix(%s)", path);
  }
  bool success = false;
  BY_HANDLE_FILE_INFORMATION info;
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ wpath.c_str(),
      /* dwDesiredAccess */ GENERIC_READ,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ OPEN_EXISTING,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  if (handle != INVALID_HANDLE_VALUE &&
      ::GetFileInformationByHandle(handle, &info)) {
    success = true;
    // TODO(laszlocsomor): use info.nFileSizeHigh after we updated total_size to
    // be u8 type.
    result->total_size = info.nFileSizeLow;
    // TODO(laszlocsomor): query the actual permissions and write in file_mode.
    result->file_mode = 0777;
    result->is_directory = (info.dwFileAttributes != INVALID_FILE_ATTRIBUTES) &&
                           (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  }
  ::CloseHandle(handle);
  return success;
#else   // !(defined(COMPILER_MSVC) || defined(__CYGWIN__))
  struct stat statst;
  if (stat(path, &statst) < 0) {
    return false;
  }
  result->total_size = statst.st_size;
  result->file_mode = statst.st_mode;
  result->is_directory = (statst.st_mode & S_IFDIR) != 0;
  return true;
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)
}

bool write_file(const char* path, unsigned int perm, const void* data,
                size_t size) {
  return blaze_util::WriteFile(data, size, path, perm);
}

bool read_file(const char* path, void* buffer, size_t size) {
  return blaze_util::ReadFile(path, buffer, size);
}

string get_cwd() { return blaze_util::GetCwd(); }

bool make_dirs(const char* path, unsigned int mode) {
#ifndef COMPILER_MSVC
  // TODO(laszlocsomor): respect `mode` on Windows/MSVC.
  mode |= S_IWUSR | S_IXUSR;
#endif  // not COMPILER_MSVC
  string spath(path);
  if (spath.empty()) {
    return true;
  }
  if (spath.back() != '/' && spath.back() != '\\') {
    spath = blaze_util::Dirname(spath);
  }
  return blaze_util::MakeDirectories(spath, mode);
}

}  // namespace devtools_ijar
