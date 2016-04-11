// Copyright 2015 The Bazel Authors. All rights reserved.
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

#include <stdio.h>
#include <windows.h>
#include <sys/cygwin.h>

#include "third_party/ijar/mapped_file.h"

#define MAX_ERROR 2048

namespace devtools_ijar {

static char errmsg[MAX_ERROR] = "";

void PrintLastError(const char* op) {
  char *message;
  DWORD err = GetLastError();
  FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER
          | FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<char *>(&message),
      0, NULL);
  snprintf(errmsg, MAX_ERROR, "%s: %s", op, message);
  LocalFree(message);
}

char* ToUnicodePath(const char* path) {
  // Add \\?\ as prefix to enable unicode path which allows path length longer
  // than 260
  int length = strlen(path) + 5;
  char* unicode_path = reinterpret_cast<char*>(malloc(length));
  snprintf(unicode_path, length, "\\\\?\\%s", path);
  return unicode_path;
}

struct MappedInputFileImpl {
  HANDLE file_;
  HANDLE mapping_;

  MappedInputFileImpl(HANDLE file, HANDLE mapping) {
    file_ = file;
    mapping_ = mapping;
  }
};

MappedInputFile::MappedInputFile(const char* name) {
  impl_ = NULL;
  opened_ = false;
  errmsg_ = errmsg;

  char* path = reinterpret_cast<char*>(
      cygwin_create_path(CCP_POSIX_TO_WIN_A, name));
  char* unicode_path = ToUnicodePath(path);
  free(path);
  HANDLE file = CreateFile(unicode_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, 0, NULL);
  free(unicode_path);
  if (file == INVALID_HANDLE_VALUE) {
    PrintLastError("CreateFile()");
    return;
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size)) {
    PrintLastError("GetFileSizeEx()");
    CloseHandle(file);
    return;
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY,
      size.HighPart, size.LowPart, NULL);
  if (mapping == NULL) {
    PrintLastError("CreateFileMapping()");
    CloseHandle(file);
    return;
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_READ, 0, 0, 0, NULL);
  if (view == NULL) {
    PrintLastError("MapViewOfFileEx()");
    CloseHandle(mapping);
    CloseHandle(file);
    return;
  }

  impl_ = new MappedInputFileImpl(file, mapping);
  length_ = size.QuadPart;
  buffer_ = reinterpret_cast<u1*>(view);
  opened_ = true;
}

MappedInputFile::~MappedInputFile() {
  delete impl_;
}

void MappedInputFile::Discard(size_t bytes) {
  // This is not supported on Windows for now. I'm not sure if we can unmap
  // parts of an existing view and that this is necessary for Windows at all.
  // At any rate, this only matters for >2GB (or maybe >4GB?) input files.
}

int MappedInputFile::Close() {
  if (!UnmapViewOfFile(buffer_)) {
    PrintLastError("UnmapViewOfFile()");
    return -1;
  }

  if (!CloseHandle(impl_->mapping_)) {
    PrintLastError("CloseHandle(mapping)");
    return -1;
  }

  if (!CloseHandle(impl_->file_)) {
    PrintLastError("CloseHandle(file)");
    return -1;
  }

  return 0;
}

struct MappedOutputFileImpl {
  HANDLE file_;
  HANDLE mapping_;

  MappedOutputFileImpl(HANDLE file, HANDLE mapping) {
    file_ = file;
    mapping_ = mapping;
  }
};

MappedOutputFile::MappedOutputFile(const char* name, u8 estimated_size) {
  impl_ = NULL;
  opened_ = false;
  errmsg_ = errmsg;

  char* path = reinterpret_cast<char*>(
      cygwin_create_path(CCP_POSIX_TO_WIN_A, name));
  char* unicode_path = ToUnicodePath(path);
  free(path);
  HANDLE file = CreateFile(unicode_path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                           CREATE_ALWAYS, 0, NULL);
  free(unicode_path);
  if (file == INVALID_HANDLE_VALUE) {
    PrintLastError("CreateFile()");
    return;
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READWRITE,
      estimated_size >> 32, estimated_size & 0xffffffffUL, NULL);
  if (mapping == NULL) {
    PrintLastError("CreateFileMapping()");
    CloseHandle(file);
    return;
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0, NULL);
  if (view == NULL) {
    PrintLastError("MapViewOfFileEx()");
    CloseHandle(mapping);
    CloseHandle(file);
    return;
  }

  impl_ = new MappedOutputFileImpl(file, mapping);
  buffer_ = reinterpret_cast<u1*>(view);
  opened_ = true;
}

MappedOutputFile::~MappedOutputFile() {
  delete impl_;
}

int MappedOutputFile::Close(int size) {
  if (!UnmapViewOfFile(buffer_)) {
    PrintLastError("UnmapViewOfFile()");
    return -1;
  }

  if (!CloseHandle(impl_->mapping_)) {
    PrintLastError("CloseHandle(mapping)");
    return -1;
  }

  if (!SetFilePointer(impl_->file_, size, NULL, FILE_BEGIN)) {
    PrintLastError("SetFilePointer()");
    return -1;
  }

  if (!SetEndOfFile(impl_->file_)) {
    PrintLastError("SetEndOfFile()");
    return -1;
  }

  if (!CloseHandle(impl_->file_)) {
    PrintLastError("CloseHandle(file)");
    return -1;
  }

  return 0;
}

}  // namespace devtools_ijar
