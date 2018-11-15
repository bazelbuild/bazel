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

#include "src/main/native/windows/util.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

namespace bazel {
namespace windows {

using std::wstring;
using std::wstringstream;

wstring MakeErrorMessage(const wchar_t* file, int line,
                         const wchar_t* failed_func, const wstring& func_arg,
                         const wstring& message) {
  wstringstream result;
  result << L"ERROR: " << file << L"(" << line << L"): " << failed_func << L"("
         << func_arg << L"): " << message;
  return result.str();
}

wstring MakeErrorMessage(const wchar_t* file, int line,
                         const wchar_t* failed_func, const wstring& func_arg,
                         DWORD error_code) {
  return MakeErrorMessage(file, line, failed_func, func_arg,
                          GetLastErrorString(error_code));
}

wstring GetLastErrorString(DWORD error_code) {
  if (error_code == 0) {
    return L"";
  }

  LPWSTR message = NULL;
  DWORD size = FormatMessageW(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
          FORMAT_MESSAGE_ALLOCATE_BUFFER,
      NULL, error_code, LANG_USER_DEFAULT, (LPWSTR)&message, 0, NULL);

  if (size == 0) {
    wstringstream err;
    DWORD format_message_error = GetLastError();
    err << L"Error code " << error_code
        << L"; cannot format message due to error code "
        << format_message_error;
    return err.str();
  }

  wstring result(message);
  HeapFree(GetProcessHeap(), LMEM_FIXED, message);
  return result;
}

bool AutoAttributeList::Create(HANDLE* handles, size_t count,
                               std::unique_ptr<AutoAttributeList>* result,
                               wstring* error_msg) {
  static constexpr DWORD kAttributeCount = 1;

  SIZE_T size = 0;
  // According to MSDN, the first call to InitializeProcThreadAttributeList is
  // expected to fail.
  InitializeProcThreadAttributeList(NULL, kAttributeCount, 0, &size);
  std::unique_ptr<uint8_t> data(new uint8_t[size]);
  LPPROC_THREAD_ATTRIBUTE_LIST attrs =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(data.get());
  if (!InitializeProcThreadAttributeList(attrs, kAttributeCount, 0, &size)) {
    if (error_msg) {
      DWORD err = GetLastError();
      *error_msg = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                    L"InitializeProcThreadAttributeList", L"",
                                    err);
    }
    return false;
  }

  if (!UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                 handles, count * sizeof(HANDLE), NULL, NULL)) {
    if (error_msg) {
      DWORD err = GetLastError();
      *error_msg = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                    L"UpdateProcThreadAttribute", L"", err);
    }
    return false;
  }
  result->reset(new AutoAttributeList(data.release()));
  return true;
}

AutoAttributeList::AutoAttributeList(uint8_t* data) : data_(data) {}

AutoAttributeList::~AutoAttributeList() {
  DeleteProcThreadAttributeList(*this);
}

AutoAttributeList::operator LPPROC_THREAD_ATTRIBUTE_LIST() const {
  return reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(data_.get());
}

static void QuotePath(const wstring& path, wstring* result) {
  *result = wstring(L"\"") + path + L"\"";
}

static bool IsSeparator(WCHAR c) { return c == L'/' || c == L'\\'; }

static bool HasSeparator(const wstring& s) {
  return s.find_first_of(L'/') != wstring::npos ||
         s.find_first_of(L'\\') != wstring::npos;
}

static bool Contains(const wstring& s, const WCHAR* substr) {
  return s.find(substr) != wstring::npos;
}

wstring AsShortPath(wstring path, wstring* result) {
  if (path.empty()) {
    result->clear();
    return L"";
  }
  if (path[0] == '"') {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"AsShortPath", path,
                            L"path should not be quoted");
  }
  if (IsSeparator(path[0])) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"AsShortPath", path,
                            L"path is absolute without a drive letter");
  }
  if (Contains(path, L"/./") || Contains(path, L"\\.\\") ||
      Contains(path, L"/..") || Contains(path, L"\\..")) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"AsShortPath", path,
                            L"path is not normalized");
  }
  if (path.size() >= MAX_PATH && !HasSeparator(path)) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"AsShortPath", path,
                            L"path is just a file name but too long");
  }
  if (HasSeparator(path) &&
      !(isalpha(path[0]) && path[1] == L':' && IsSeparator(path[2]))) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"AsShortPath", path,
                            L"path is not absolute");
  }
  // At this point we know the path is either just a file name (shorter than
  // MAX_PATH), or an absolute, normalized, Windows-style path (of any length).

  std::replace(path.begin(), path.end(), '/', '\\');
  // Fast-track: the path is already short.
  if (path.size() < MAX_PATH) {
    *result = path;
    return L"";
  }
  // At this point we know that the path is at least MAX_PATH long and that it's
  // absolute, normalized, and Windows-style.

  wstring wlong = wstring(L"\\\\?\\") + path;

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
    DWORD err_code = GetLastError();
    wstring res = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                   L"GetShortPathNameW", wlong, err_code);
    return res;
  }

  if (wshort_size >= kMaxShortPath) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__, L"GetShortPathNameW",
                            wlong, L"cannot shorten the path enough");
  }
  GetShortPathNameW(wlong.c_str(), wshort, kMaxShortPath);
  result->assign(wshort + 4);
  return L"";
}

wstring AsExecutablePathForCreateProcess(const wstring& path, wstring* result) {
  if (path.empty()) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__,
                            L"AsExecutablePathForCreateProcess", path,
                            L"path should not be empty");
  }
  wstring error = AsShortPath(path, result);
  if (!error.empty()) {
    return MakeErrorMessage(WSTR(__FILE__), __LINE__,
                            L"AsExecutablePathForCreateProcess", path, error);
  }
  // Quote the path in case it's something like "c:\foo\app name.exe".
  // Do this unconditionally, there's no harm in quoting. Quotes are not
  // allowed inside paths so we don't need to escape quotes.
  QuotePath(*result, result);
  return L"";
}

}  // namespace windows
}  // namespace bazel
