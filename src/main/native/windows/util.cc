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
#include <stdlib.h>
#include <windows.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

using std::function;
using std::string;
using std::unique_ptr;
using std::wstring;

string GetLastErrorString(const string& cause) {
  DWORD last_error = GetLastError();
  if (last_error == 0) {
    return "";
  }

  LPSTR message;
  DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, last_error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&message, 0, NULL);

  if (size == 0) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s: Error %d (cannot format message due to error %d)",
             cause.c_str(), last_error, GetLastError());
    buf[sizeof(buf) - 1] = 0;
  }

  string result = string(message);
  LocalFree(message);
  return cause + ": " + result;
}

static void QuotePath(const string& path, string* result) {
  *result = string("\"") + path + "\"";
}

static bool IsSeparator(char c) { return c == '/' || c == '\\'; }

static bool HasSeparator(const string& s) {
  return s.find_first_of('/') != string::npos ||
         s.find_first_of('\\') != string::npos;
}

static bool Contains(const string& s, const char* substr) {
  return s.find(substr) != string::npos;
}

string AsShortPath(string path, function<wstring()> path_as_wstring,
                   string* result) {
  if (path.empty()) {
    result->clear();
    return "";
  }
  if (path[0] == '"') {
    return string("path should not be quoted");
  }
  if (IsSeparator(path[0])) {
    return string("path='") + path + "' is absolute";
  }
  if (Contains(path, "/./") || Contains(path, "\\.\\") ||
      Contains(path, "/..") || Contains(path, "\\..")) {
    return string("path='") + path + "' is not normalized";
  }
  if (path.size() >= MAX_PATH && !HasSeparator(path)) {
    return string("path='") + path + "' is just a file name but too long";
  }
  if (HasSeparator(path) &&
      !(isalpha(path[0]) && path[1] == ':' && IsSeparator(path[2]))) {
    return string("path='") + path + "' is not an absolute path";
  }
  // At this point we know the path is either just a file name (shorter than
  // MAX_PATH), or an absolute, normalized, Windows-style path (of any length).

  std::replace(path.begin(), path.end(), '/', '\\');
  // Fast-track: the path is already short.
  if (path.size() < MAX_PATH) {
    *result = path;
    return "";
  }
  // At this point we know that the path is at least MAX_PATH long and that it's
  // absolute, normalized, and Windows-style.

  // Retrieve string as UTF-16 path, add "\\?\" prefix.
  wstring wlong = wstring(L"\\\\?\\") + path_as_wstring();

  // Experience shows that:
  // - GetShortPathNameW's result has a "\\?\" prefix if and only if the input
  //   did too (though this behavior is not documented on MSDN)
  // - CreateProcess{A,W} only accept an executable of MAX_PATH - 1 length
  // Therefore for our purposes the acceptable shortened length is
  // MAX_PATH + 4 (null-terminated). That is, MAX_PATH - 1 for the shortened
  // path, plus a potential "\\?\" prefix that's only there if `wlong` also had
  // it and which we'll omit from `result`, plus a null terminator.
  static const size_t kMaxShortPath = MAX_PATH + 4;

  WCHAR wshort[kMaxShortPath];
  DWORD wshort_size = ::GetShortPathNameW(wlong.c_str(), NULL, 0);
  if (wshort_size == 0) {
    return GetLastErrorString(string("GetShortPathName failed (path=") + path +
                              ")");
  }

  if (wshort_size >= kMaxShortPath) {
    return string("GetShortPathName would not shorten the path enough (path=") +
           path + ")";
  }
  GetShortPathNameW(wlong.c_str(), wshort, kMaxShortPath);

  // Convert the result to UTF-8.
  char mbs_short[MAX_PATH];
  size_t mbs_size = wcstombs(
      mbs_short,
      wshort + 4,  // we know it has a "\\?\" prefix, because `wlong` also did
      MAX_PATH);
  if (mbs_size < 0 || mbs_size >= MAX_PATH) {
    return string("wcstombs failed (path=") + path + ")";
  }
  mbs_short[mbs_size] = 0;

  *result = mbs_short;
  return "";
}

string AsExecutablePathForCreateProcess(const string& path,
                                        function<wstring()> path_as_wstring,
                                        string* result) {
  if (path.empty()) {
    return string("path should not be empty");
  }
  string error = AsShortPath(path, path_as_wstring, result);
  if (error.empty()) {
    // Quote the path in case it's something like "c:\foo\app name.exe".
    // Do this unconditionally, there's no harm in quoting. Quotes are not
    // allowed inside paths so we don't need to escape quotes.
    QuotePath(*result, result);
  }
  return error;
}

}  // namespace windows
}  // namespace bazel
