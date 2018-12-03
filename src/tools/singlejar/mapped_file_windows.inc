// Copyright 2018 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_ 1

#if !defined(_WIN64)
#error This code is for 64 bit Windows.
#endif

#include "src/tools/singlejar/diag.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <string>

MappedFile::MappedFile()
    : mapped_start_(nullptr),
      mapped_end_(nullptr),
      fd_(-1),
      hMapFile_(INVALID_HANDLE_VALUE) {}

bool MappedFile::Open(const std::string& path) {
  if (is_open()) {
    diag_errx(1, "%s:%d: This instance is already open", __FILE__, __LINE__);
  }
  if ((fd_ = _open(path.c_str(), O_RDONLY | O_BINARY)) < 0) {
    diag_warn("%s:%d: open %s:", __FILE__, __LINE__, path.c_str());
    return false;
  }

  HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
  LARGE_INTEGER temp;
  ::GetFileSizeEx(hFile, &temp);
  size_t fileSize = temp.QuadPart;

  if (fileSize == 0) {
    // Handle empty files specially, because CreateFileMapping cannot map them:
    //
    // From
    // https://docs.microsoft.com/en-us/windows/desktop/api/winbase/nf-winbase-createfilemappinga
    //
    // An attempt to map a file with a length of 0 (zero) fails with an error
    // code of ERROR_FILE_INVALID. Applications should test for files with a
    // length of 0 (zero) and reject those files.
    mapped_start_ = nullptr;
    mapped_end_ = nullptr;
    hMapFile_ = INVALID_HANDLE_VALUE;
    return true;
  }

  hMapFile_ = ::CreateFileMapping(
      hFile,
      nullptr,                             // default security
      PAGE_READONLY,                       // read-only permission
      static_cast<DWORD>(fileSize >> 32),  // size of mapping object, high
      static_cast<DWORD>(fileSize),        // size of mapping object, low
      nullptr);                            // name of mapping object

  if (hMapFile_ == nullptr) {
    diag_warn("%s:%d: CreateFileMapping for %s failed", __FILE__, __LINE__,
              path.c_str());
    _close(fd_);
    fd_ = -1;
    return false;
  }

  mapped_start_ = static_cast<unsigned char*>(
      MapViewOfFile(hMapFile_,
                    FILE_MAP_READ | FILE_MAP_COPY,  // PROT_READ | MAP_PRIVATE
                    0,                              // file offset, high
                    0,                              // file offset, low
                    fileSize));                     // file size
  if (mapped_start_ == nullptr) {
    diag_warn("%s:%d: MapViewOfFile for %s failed", __FILE__, __LINE__,
              path.c_str());
    ::CloseHandle(hMapFile_);
    _close(fd_);
    hMapFile_ = INVALID_HANDLE_VALUE;
    fd_ = -1;
    return false;
  }

  mapped_end_ = mapped_start_ + fileSize;
  return true;
}

void MappedFile::Close() {
  if (is_open()) {
    if (mapped_start_) {
      ::UnmapViewOfFile(mapped_start_);
    }
    if (hMapFile_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(hMapFile_);
      hMapFile_ = INVALID_HANDLE_VALUE;
    }
    _close(fd_);
    fd_ = -1;
    mapped_start_ = mapped_end_ = nullptr;
  }
}

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_
