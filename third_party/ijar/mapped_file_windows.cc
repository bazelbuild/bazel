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

#include <string>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file_platform.h"
#include "third_party/ijar/mapped_file.h"

#define MAX_ERROR 2048

namespace devtools_ijar {

using std::wstring;

static char errmsg[MAX_ERROR] = "";

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

  wstring wname;
  if (!blaze_util::AsAbsoluteWindowsPath(name, &wname)) {
    blaze_util::pdie(255, "MappedInputFile(%s): AsAbsoluteWindowsPath", name);
  }
  HANDLE file = CreateFileW(wname.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, 0, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    blaze_util::pdie(255, "MappedInputFile(%s): CreateFileW(%S)", name,
                     wname.c_str());
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size)) {
    blaze_util::pdie(255, "MappedInputFile(%s): GetFileSizeEx", name);
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY,
      size.HighPart, size.LowPart, NULL);
  if (mapping == NULL || mapping == INVALID_HANDLE_VALUE) {
    blaze_util::pdie(255, "MappedInputFile(%s): CreateFileMapping", name);
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_READ, 0, 0, 0, NULL);
  if (view == NULL) {
    blaze_util::pdie(255, "MappedInputFile(%s): MapViewOfFileEx", name);
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
    blaze_util::pdie(255, "MappedInputFile::Close: UnmapViewOfFile");
  }

  if (!CloseHandle(impl_->mapping_)) {
    blaze_util::pdie(255, "MappedInputFile::Close: CloseHandle for mapping");
  }

  if (!CloseHandle(impl_->file_)) {
    blaze_util::pdie(255, "MappedInputFile::Close: CloseHandle for file");
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

  wstring wname;
  if (!blaze_util::AsAbsoluteWindowsPath(name, &wname)) {
    blaze_util::pdie(255, "MappedOutputFile(%s): AsAbsoluteWindowsPath", name);
  }
  HANDLE file = CreateFileW(wname.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                            NULL, CREATE_ALWAYS, 0, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    blaze_util::pdie(255, "MappedOutputFile(%s): CreateFileW(%S)", name,
                     wname.c_str());
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READWRITE,
      estimated_size >> 32, estimated_size & 0xffffffffUL, NULL);
  if (mapping == NULL || mapping == INVALID_HANDLE_VALUE) {
    blaze_util::pdie(255, "MappedOutputFile(%s): CreateFileMapping", name);
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0, NULL);
  if (view == NULL) {
    blaze_util::pdie(255, "MappedOutputFile(%s): MapViewOfFileEx", name);
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
    blaze_util::pdie(255, "MappedOutputFile::Close: UnmapViewOfFile");
  }

  if (!CloseHandle(impl_->mapping_)) {
    blaze_util::pdie(255, "MappedOutputFile::Close: CloseHandle for mapping");
  }

  if (!SetFilePointer(impl_->file_, size, NULL, FILE_BEGIN)) {
    blaze_util::pdie(255, "MappedOutputFile::Close: SetFilePointer");
  }

  if (!SetEndOfFile(impl_->file_)) {
    blaze_util::pdie(255, "MappedOutputFile::Close: SetEndOfFile");
  }

  if (!CloseHandle(impl_->file_)) {
    blaze_util::pdie(255, "MappedOutputFile::Close: CloseHandle for file");
  }

  return 0;
}

}  // namespace devtools_ijar
