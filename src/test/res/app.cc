// Copyright 2019 The Bazel Authors. All rights reserved.
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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "src/test/res/app.h"
#endif  // _WIN32

int main() {
#ifdef _WIN32
  WCHAR p[100];
  memset(p, 0, sizeof(p));
  int l = LoadStringW(GetModuleHandle(nullptr), IDS_STRING, p, 100);
  wprintf(L"l=%d, p=(%s)", l, p);
#else   // not _WIN32
  printf("not supported");
#endif  // _WIN32
  return 0;
}
