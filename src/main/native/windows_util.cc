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

#include <functional>
#include <memory>
#include <string>

#include "src/main/native/windows_util.h"

namespace windows_util {

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

  string result = string(message);
  LocalFree(message);
  return cause + ": " + result;
}

static void QuotePath(const string& path, string* result) {
  *result = string("\"") + path + "\"";
}

string AsExecutablePathForCreateProcess(const string& path,
                                        function<wstring()> path_as_wstring,
                                        string* result) {
  if (path.empty()) {
    return string("argv[0] should not be empty");
  }
  if (path[0] == '"') {
    return string("argv[0] should not be quoted");
  }
  if (path[0] == '\\' ||                 // absolute, but without drive letter
      path.find("/") != string::npos ||  // has "/"
      path.find("\\.\\") != string::npos ||   // not normalized
      path.find("\\..\\") != string::npos ||  // not normalized
      // at least MAX_PATH long, but just a file name
      (path.size() >= MAX_PATH && path.find_first_of('\\') == string::npos) ||
      // not just a file name, but also not absolute
      (path.find_first_of('\\') != string::npos &&
       !(isalpha(path[0]) && path[1] == ':' && path[2] == '\\'))) {
    return string("argv[0]='" + path +
                  "'; should have been either an absolute, "
                  "normalized, Windows-style path with drive letter (e.g. "
                  "'c:\\foo\\bar.exe'), or just a file name (e.g. "
                  "'cmd.exe') shorter than MAX_PATH.");
  }
  // At this point we know the path is either just a file name (shorter than
  // MAX_PATH), or an absolute, normalized, Windows-style path (of any length).

  // Fast-track: the path is already short.
  if (path.size() < MAX_PATH) {
    // Quote the path in case it's something like "c:\foo\app name.exe".
    // Do this unconditionally, there's no harm in quoting. Quotes are not
    // allowed inside paths so we don't need to escape quotes.
    QuotePath(path, result);
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
    return windows_util::GetLastErrorString(
        string("GetShortPathName failed (path=") + path + ")");
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

  QuotePath(mbs_short, result);
  return "";
}

}  // namespace windows_util
