// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <sstream>
#include <string>

#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::ostringstream;
using std::string;
using std::stringstream;

string GetLastErrorString() {
  DWORD last_error = GetLastError();
  if (last_error == 0) {
    return string();
  }

  char* message_buffer;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, last_error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&message_buffer, 0, NULL);

  stringstream result;
  result << "(error: " << last_error << "): " << message_buffer;
  LocalFree(message_buffer);
  return result.str();
}

void die(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  fputs("LAUNCHER ERROR: ", stderr);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

void PrintError(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  fputs("LAUNCHER ERROR: ", stderr);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
}

bool DoesFilePathExist(const char* path) {
  // TODO(laszlocsomor): convert `path` to (const wchar_t*), add longpath-prefix
  // and use GetFileAttributesW.
  DWORD dwAttrib = GetFileAttributesA(path);

  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

string GetBinaryPathWithoutExtension(const string& binary) {
  if (binary.find(".exe", binary.size() - 4) != string::npos) {
    return binary.substr(0, binary.length() - 4);
  }
  return binary;
}

string GetBinaryPathWithExtension(const string& binary) {
  return GetBinaryPathWithoutExtension(binary) + ".exe";
}

string GetEscapedArgument(const string& argument) {
  ostringstream escaped_arg;
  bool has_space = argument.find_first_of(' ') != string::npos;

  if (has_space) {
    escaped_arg << '\"';
  }

  string::const_iterator it = argument.begin();
  while (it != argument.end()) {
    char ch = *it++;
    switch (ch) {
      case '"':
        // Escape double quotes
        escaped_arg << "\\\"";
        break;

      case '\\':
        // Escape back slashes
        escaped_arg << "\\\\";
        break;

      default:
        escaped_arg << ch;
    }
  }

  if (has_space) {
    escaped_arg << '\"';
  }
  return escaped_arg.str();
}

}  // namespace launcher
}  // namespace bazel
