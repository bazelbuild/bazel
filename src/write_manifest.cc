// Copyright 2024 The Bazel Authors. All rights reserved.
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
#include <stdlib.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string>

// Writes a new app manifest to a Windows executable.
int wmain(int argc, wchar_t *argv[]) {
  if (argc != 2) {
    fwprintf(stderr, L"Usage: %ls <filename>\n", argv[0]);
    return 1;
  }

  std::string new_manifest;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) {
    new_manifest.append(buf, n);
  }

  fwprintf(stderr, L"%hs\n", new_manifest.c_str());

  HANDLE update_handle = BeginUpdateResourceW(argv[1], false);
  if (!update_handle) {
    fwprintf(stderr, L"Error opening file %ls for update: %d\n", argv[1],
             GetLastError());
    return 1;
  }
  if (!UpdateResourceA(update_handle, RT_MANIFEST, MAKEINTRESOURCE(1),
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL),
                       const_cast<char *>(new_manifest.c_str()),
                       new_manifest.size())) {
    fwprintf(stderr, L"Error updating resource: %d\n", GetLastError());
    return 1;
  }
  if (!EndUpdateResourceW(update_handle, false)) {
    fwprintf(stderr, L"Error finalizing update: %d\n", GetLastError());
    return 1;
  }

  return 0;
}
