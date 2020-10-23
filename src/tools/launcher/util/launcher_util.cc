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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// For rand_s function, https://msdn.microsoft.com/en-us/library/sxtz2fa8.aspx
#define _CRT_RAND_S
#include <fcntl.h>
#include <io.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <sstream>
#include <string>

#include "src/main/cpp/util/path_platform.h"
#include "src/main/native/windows/file.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::string;
using std::stringstream;
using std::wostringstream;
using std::wstring;

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

void die(const wchar_t* format, ...) {
  // Set translation mode to _O_U8TEXT so that we can display
  // error message containing unicode correctly.
  _setmode(_fileno(stderr), _O_U8TEXT);
  va_list ap;
  va_start(ap, format);
  fputws(L"LAUNCHER ERROR: ", stderr);
  vfwprintf(stderr, format, ap);
  va_end(ap);
  fputwc(L'\n', stderr);
  exit(1);
}

void PrintError(const wchar_t* format, ...) {
  // Set translation mode to _O_U8TEXT so that we can display
  // error message containing unicode correctly.
  // _setmode returns -1 if it fails to set the mode.
  int previous_mode = _setmode(_fileno(stderr), _O_U8TEXT);
  va_list ap;
  va_start(ap, format);
  fputws(L"LAUNCHER ERROR: ", stderr);
  vfwprintf(stderr, format, ap);
  va_end(ap);
  fputwc(L'\n', stderr);
  // Set translation mode back to the original one if it's changed.
  if (previous_mode != -1) {
    _setmode(_fileno(stderr), previous_mode);
  }
}

wstring AsAbsoluteWindowsPath(const wchar_t* path) {
  wstring wpath;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(path, &wpath, &error)) {
    die(L"Couldn't convert %s to absolute Windows path: %hs", path,
        error.c_str());
  }
  return wpath;
}

bool DoesFilePathExist(const wchar_t* path) {
  DWORD dwAttrib = GetFileAttributesW(AsAbsoluteWindowsPath(path).c_str());

  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool DoesDirectoryPathExist(const wchar_t* path) {
  DWORD dwAttrib = GetFileAttributesW(AsAbsoluteWindowsPath(path).c_str());

  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

bool DeleteFileByPath(const wchar_t* path) {
  return DeleteFileW(AsAbsoluteWindowsPath(path).c_str());
}

bool DeleteDirectoryByPath(const wchar_t* path) {
  return RemoveDirectoryW(AsAbsoluteWindowsPath(path).c_str());
}

static bool EndsWithExe(const wstring& s) {
  return s.size() >= 4 && _wcsnicmp(s.c_str() + s.size() - 4, L".exe", 4) == 0;
}

wstring GetWindowsLongPath(const wstring& path) {
  std::unique_ptr<WCHAR[]> result;
  wstring abs_path = AsAbsoluteWindowsPath(path.c_str());
  wstring error = bazel::windows::GetLongPath(abs_path.c_str(), &result);
  if (!error.empty()) {
    PrintError(L"Failed to get long path for %s: %s", path.c_str(),
               error.c_str());
  }
  return wstring(blaze_util::RemoveUncPrefixMaybe(result.get()));
}

wstring GetBinaryPathWithoutExtension(const wstring& binary) {
  return EndsWithExe(binary) ? binary.substr(0, binary.size() - 4) : binary;
}

wstring GetBinaryPathWithExtension(const wstring& binary) {
  return EndsWithExe(binary) ? binary : (binary + L".exe");
}

std::wstring BashEscapeArg(const std::wstring& argument) {
  wstring escaped_arg;
  // escaped_arg will be at least this long
  escaped_arg.reserve(argument.size());
  bool has_space = argument.find_first_of(L' ') != wstring::npos;

  if (argument.empty()) {
    return L"\"\"";
  }

  if (has_space) {
    escaped_arg += L'\"';
  }

  for (const wchar_t ch : argument) {
    switch (ch) {
      case L'"':
        // Escape double quotes
        escaped_arg += L"\\\"";
        break;

      case L'\\':
        // Escape back slashes.
        escaped_arg += L"\\\\";
        break;

      default:
        escaped_arg += ch;
    }
  }

  if (has_space) {
    escaped_arg += L'\"';
  }
  return escaped_arg;
}

// An environment variable has a maximum size limit of 32,767 characters
// https://msdn.microsoft.com/en-us/library/ms683188.aspx
static const int BUFFER_SIZE = 32767;

bool GetEnv(const wstring& env_name, wstring* value) {
  wchar_t buffer[BUFFER_SIZE];
  if (!GetEnvironmentVariableW(env_name.c_str(), buffer, BUFFER_SIZE)) {
    return false;
  }
  *value = buffer;
  return true;
}

bool SetEnv(const wstring& env_name, const wstring& value) {
  return SetEnvironmentVariableW(env_name.c_str(), value.c_str());
}

wstring GetRandomStr(size_t len) {
  static const wchar_t alphabet[] =
      L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  wstring rand_str;
  rand_str.reserve(len);
  unsigned int x;
  for (size_t i = 0; i < len; i++) {
    rand_s(&x);
    rand_str += alphabet[x % wcslen(alphabet)];
  }
  return rand_str;
}

bool NormalizePath(const wstring& path, wstring* result) {
  string error;
  if (!blaze_util::AsWindowsPath(path, result, &error)) {
    PrintError(L"Failed to normalize %s: %hs", path.c_str(), error.c_str());
    return false;
  }
  std::transform(result->begin(), result->end(), result->begin(), ::tolower);
  return true;
}

wstring GetBaseNameFromPath(const wstring& path) {
  return path.substr(path.find_last_of(L"\\/") + 1);
}

wstring GetParentDirFromPath(const wstring& path) {
  return path.substr(0, path.find_last_of(L"\\/"));
}

bool RelativeTo(const wstring& path, const wstring& base, wstring* result) {
  if (blaze_util::IsAbsolute(path) != blaze_util::IsAbsolute(base)) {
    PrintError(
        L"Cannot calculate relative path from an absolute and a non-absolute"
        " path.\npath = %s\nbase = %s",
        path.c_str(), base.c_str());
    return false;
  }

  if (blaze_util::IsAbsolute(path) && blaze_util::IsAbsolute(base) &&
      path[0] != base[0]) {
    PrintError(
        L"Cannot calculate relative path from absolute path under different "
        "drives."
        "\npath = %s\nbase = %s",
        path.c_str(), base.c_str());
    return false;
  }

  // Record the back slash position after the last matched path fragment
  int pos = 0;
  int back_slash_pos = -1;
  while (path[pos] == base[pos] && base[pos] != L'\0') {
    if (path[pos] == L'\\') {
      back_slash_pos = pos;
    }
    pos++;
  }

  if (base[pos] == L'\0' && path[pos] == L'\0') {
    // base == path in this case
    result->assign(L"");
    return true;
  }

  if ((base[pos] == L'\0' && path[pos] == L'\\') ||
      (base[pos] == L'\\' && path[pos] == L'\0')) {
    // In this case, one of the paths is the parent of another one.
    // We should move back_slash_pos to the end of the shorter path.
    // eg. path = c:\foo\bar, base = c:\foo => back_slash_pos = 6
    //  or path = c:\foo, base = c:\foo\bar => back_slash_pos = 6
    back_slash_pos = pos;
  }

  wostringstream buffer;

  // Create the ..\\ prefix
  // eg. path = C:\foo\bar1, base = C:\foo\bar2, we need ..\ prefix
  // In case "base" is a parent of "path", we set back_slash_pos to the end
  // of "base", so we need no prefix when back_slash_pos + 1 > base.length().
  // back_slash_pos + 1 == base.length() is not possible because the last
  // character of a normalized path won't be back slash.
  if (back_slash_pos + 1 < base.length()) {
    buffer << L"..\\";
  }
  for (int i = back_slash_pos + 1; i < base.length(); i++) {
    if (base[i] == L'\\') {
      buffer << L"..\\";
    }
  }

  // Add the result of not matched path fragments into result
  // eg. path = C:\foo\bar1, base = C:\foo\bar2, adding `bar1`
  // In case "path" is a parent of "base", we set back_slash_pos to the end
  // of "path", so we need no suffix when back_slash_pos == path.length().
  if (back_slash_pos != path.length()) {
    buffer << path.substr(back_slash_pos + 1);
  }

  result->assign(buffer.str());
  return true;
}

}  // namespace launcher
}  // namespace bazel
