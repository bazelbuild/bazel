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

// Extracts the app manifest of a Windows executable and prints it to stdout.
int wmain(int argc, wchar_t *argv[]) {
  if (argc != 2) {
    fwprintf(stderr, L"Usage: %ls <filename>\n", argv[0]);
    return 1;
  }

  // Read the app manifest (aka side-by-side or fusion manifest) from the
  // executable, which requires loading it as a "module".
  HMODULE exe = LoadLibraryExW(argv[1], nullptr, LOAD_LIBRARY_AS_DATAFILE);
  if (!exe) {
    fwprintf(stderr, L"Error loading file %ls: %d\n", argv[1], GetLastError());
    return 1;
  }
  HRSRC manifest_resource = FindResourceA(exe, MAKEINTRESOURCE(1), RT_MANIFEST);
  if (!manifest_resource) {
    fwprintf(stderr, L"Resource not found: %d\n", GetLastError());
    return 1;
  }
  HGLOBAL manifest_handle = LoadResource(exe, manifest_resource);
  if (!manifest_handle) {
    fwprintf(stderr, L"Error loading resource: %d\n", GetLastError());
    return 1;
  }
  LPVOID manifest_data = LockResource(manifest_handle);
  if (!manifest_data) {
    fwprintf(stderr, L"Error locking resource: %d\n", GetLastError());
    return 1;
  }
  DWORD manifest_len = SizeofResource(exe, manifest_resource);

  // Write the manifest to stdout.
  fwrite(manifest_data, 1, manifest_len, stdout);

  UnlockResource(manifest_handle);
  FreeResource(manifest_handle);
  FreeLibrary(exe);

  return 0;
}
