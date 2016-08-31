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

#include <stdio.h>
#include <windows.h>

#include <string>

#include "src/main/native/windows_error_handling.h"

std::string GetLastErrorString(const std::string& cause) {
  DWORD last_error = GetLastError();
  if (last_error == 0) {
    return "";
  }

  LPSTR message;
  DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER
          | FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      last_error,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR) &message,
      0,
      NULL);

  if (size == 0) {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "%s: Error %d (cannot format message due to error %d)",
        cause.c_str(), last_error, GetLastError());
    buf[sizeof(buf) - 1] = 0;
  }

  std::string result = std::string(message);
  LocalFree(message);
  return cause + ": " + result;
}
