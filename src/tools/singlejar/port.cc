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

#ifdef _WIN32

#include "src/tools/singlejar/port.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

ptrdiff_t pread(int fd, void *buf, size_t count, ptrdiff_t offset) {
  DWORD ret = -1;
  HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (hFile) {
    OVERLAPPED overlap = {0};
    overlap.Offset = offset;
    ::ReadFile(hFile, buf, count, &ret, &overlap);
  }
  return static_cast<int>(ret);
}

#endif  // _WIN32
