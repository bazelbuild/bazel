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

#include <errno.h>
#include <limits.h>
#include <stdio.h>

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
#include <windows.h>
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

#ifndef COMPILER_MSVC
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // not COMPILER_MSVC

#include <string>

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file_platform.h"
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

namespace devtools_ijar {

using std::string;

bool stat_file(const char* path, Stat* result) {
#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
  std::wstring wpath;
  if (!blaze_util::AsWindowsPathWithUncPrefix(path, &wpath)) {
    blaze_util::die(255, "stat_file: AsWindowsPathWithUncPrefix(%s) failed",
                    path);
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

bool write_file(const char* path, mode_t perm, const void* data, size_t size) {
#ifdef COMPILER_MSVC
  // TODO(laszlocsomor) 2016-12-01: implement this and other methods, in order
  // to close https://github.com/bazelbuild/bazel/issues/2157.
  fprintf(stderr, "Not yet implemented on Windows\n");
  return false;
#else   // not COMPILER_MSVC
  int fd = open(path, O_CREAT | O_WRONLY, perm);
  if (fd < 0) {
    fprintf(stderr, "Cannot open file %s for writing: %s\n", path,
            strerror(errno));
    return false;
  }
  bool result = true;
  int error = write(fd, data, size);
  // Check for an error condition, or if we didn't write all of the data.
  if (error < 0 || static_cast<size_t>(error) != size) {
    fprintf(stderr, "Cannot write %zu bytes to file %s: %s\n", size, path,
            strerror(errno));
    result = false;
  }
  if (close(fd)) {
    fprintf(stderr, "Cannot close file %s: %s\n", path, strerror(errno));
    result = false;
  }
  return result;
#endif  // COMPILER_MSVC
}

bool read_file(const char* path, void* buffer, size_t size) {
#ifdef COMPILER_MSVC
  // TODO(laszlocsomor) 2016-12-01: implement this and other methods, in order
  // to close https://github.com/bazelbuild/bazel/issues/2157.
  fprintf(stderr, "Not yet implemented on Windows\n");
  return false;
#else   // not COMPILER_MSVC
  // read the input file
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Can't open file %s for reading: %s\n", path,
            strerror(errno));
    return false;
  }
  bool result = true;
  size_t nb_read = 0;
  while (nb_read < size) {
    size_t to_read = size - nb_read;
    if (to_read > 16384 /* 16K */) {
      to_read = 16384;
    }
    ssize_t r = read(fd, static_cast<uint8_t*>(buffer) + nb_read, to_read);
    if (r < 0) {
      fprintf(stderr, "Can't read %zu bytes from file %s: %s\n", to_read, path,
              strerror(errno));
      result = false;
      break;
    }
    nb_read += r;
  }
  if (close(fd)) {
    fprintf(stderr, "Cannot close file %s: %s\n", path, strerror(errno));
    result = false;
  }
  return result;
#endif  // COMPILER_MSVC
}

string get_cwd() {
#ifdef COMPILER_MSVC
  // TODO(laszlocsomor) 2016-12-01: implement this and other methods, in order
  // to close https://github.com/bazelbuild/bazel/issues/2157.
  fprintf(stderr, "Not yet implemented on Windows\n");
  return "";
#else   // not COMPILER_MSVC
  char cwd[PATH_MAX];
  if (getcwd(cwd, PATH_MAX) == NULL) {
    fprintf(stderr, "getcwd() failed: %s\n", strerror(errno));
    return "";
  } else {
    return string(cwd);
  }
#endif  // COMPILER_MSVC
}

bool make_dirs(const char* path, mode_t mode) {
#ifdef COMPILER_MSVC
  // TODO(laszlocsomor) 2016-12-01: implement this and other methods, in order
  // to close https://github.com/bazelbuild/bazel/issues/2157.
  fprintf(stderr, "Not yet implemented on Windows\n");
  return false;
#else   // not COMPILER_MSVC
  // Directories created must have executable bit set and be owner writeable.
  // Otherwise, we cannot write or create any file inside.
  mode |= S_IWUSR | S_IXUSR;
  char path_[PATH_MAX];
  Stat file_stat;
  strncpy(path_, path, PATH_MAX);
  path_[PATH_MAX - 1] = 0;
  char* pointer = path_;
  while ((pointer = strchr(pointer, '/')) != NULL) {
    if (path_ != pointer) {  // skip leading slash
      *pointer = 0;
      if (!stat_file(path_, &file_stat) && mkdir(path_, mode) < 0) {
        fprintf(stderr, "Cannot create folder %s: %s\n", path_,
                strerror(errno));
        return false;
      }
      *pointer = '/';
    }
    pointer++;
  }
  return true;
#endif  // COMPILER_MSVC
}

}  // namespace devtools_ijar
