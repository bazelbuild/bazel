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

#include <windows.h>

#include <memory>

#include "src/main/native/windows_file_operations.h"
#include "src/main/native/windows_util.h"

namespace windows_util {

using std::unique_ptr;

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

bool GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  DWORD size = ::GetLongPathNameW(path, NULL, 0);
  if (size == 0) {
    return false;
  }
  result->reset(new WCHAR[size]);
  ::GetLongPathNameW(path, result->get(), size);
  return true;
}

}  // namespace windows_util
