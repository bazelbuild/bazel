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
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "third_party/ijar/mapped_file.h"

#define MAX_ERROR 2048

namespace devtools_ijar {

using std::string;
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
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(name, &wname, &error)) {
    BAZEL_DIE(255) << "MappedInputFile(" << name
                   << "): AsAbsoluteWindowsPath failed: " << error;
  }
  HANDLE file = CreateFileW(wname.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, 0, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile(" << name << "): CreateFileW("
                   << blaze_util::WstringToCstring(wname)
                   << ") failed: " << errormsg;
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size)) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile(" << name
                   << "): GetFileSizeEx failed: " << errormsg;
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY,
      size.HighPart, size.LowPart, NULL);
  if (mapping == NULL || mapping == INVALID_HANDLE_VALUE) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile(" << name
                   << "): CreateFileMapping failed: " << errormsg;
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_READ, 0, 0, 0, NULL);
  if (view == NULL) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile(" << name
                   << "): MapViewOfFileEx failed: " << errormsg;
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
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile::Close: UnmapViewOfFile failed: "
                   << errormsg;
  }

  if (!CloseHandle(impl_->mapping_)) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile::Close: CloseHandle for mapping failed: "
                   << errormsg;
  }

  if (!CloseHandle(impl_->file_)) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedInputFile::Close: CloseHandle for file failed: "
                   << errormsg;
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

MappedOutputFile::MappedOutputFile(const char* name, size_t estimated_size) {
  impl_ = NULL;
  opened_ = false;
  errmsg_ = errmsg;

  wstring wname;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(name, &wname, &error)) {
    BAZEL_DIE(255) << "MappedOutputFile(" << name
                   << "): AsAbsoluteWindowsPath failed: " << error;
  }
  HANDLE file = CreateFileW(wname.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                            NULL, CREATE_ALWAYS, 0, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedOutputFile(" << name << "): CreateFileW("
                   << blaze_util::WstringToCstring(wname)
                   << ") failed: " << errormsg;
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READWRITE,
      estimated_size >> 32, estimated_size & 0xffffffffUL, NULL);
  if (mapping == NULL || mapping == INVALID_HANDLE_VALUE) {
    BAZEL_DIE(255) << "MappedOutputFile(" << name
                   << "): CreateFileMapping failed";
  }

  void *view = MapViewOfFileEx(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0, NULL);
  if (view == NULL) {
    string errormsg = blaze_util::GetLastErrorString();
    BAZEL_DIE(255) << "MappedOutputFile(" << name
                   << "): MapViewOfFileEx failed: " << errormsg;
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

int MappedOutputFile::Close(size_t size) {
  if (!UnmapViewOfFile(buffer_)) {
    BAZEL_DIE(255) << "MappedOutputFile::Close: UnmapViewOfFile failed: "
                   << blaze_util::GetLastErrorString();
  }

  if (!CloseHandle(impl_->mapping_)) {
    BAZEL_DIE(255)
        << "MappedOutputFile::Close: CloseHandle for mapping failed: "
        << blaze_util::GetLastErrorString();
  }

  if (!SetFilePointer(impl_->file_, size, NULL, FILE_BEGIN)) {
    BAZEL_DIE(255) << "MappedOutputFile::Close: SetFilePointer failed: "
                   << blaze_util::GetLastErrorString();
  }

  if (!SetEndOfFile(impl_->file_)) {
    BAZEL_DIE(255) << "MappedOutputFile::Close: SetEndOfFile failed: "
                   << blaze_util::GetLastErrorString();
  }

  if (!CloseHandle(impl_->file_)) {
    BAZEL_DIE(255) << "MappedOutputFile::Close: CloseHandle for file failed: "
                   << blaze_util::GetLastErrorString();
  }

  return 0;
}

}  // namespace devtools_ijar
