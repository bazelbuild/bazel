// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: laszlocsomor@google.com (Laszlo Csomor)
//
// Implementation for long-path-aware open/mkdir/access on Windows.
//
// These functions convert the input path to an absolute Windows path
// with UNC prefix if necessary, then pass that to
// _wopen/_wmkdir/_waccess (declared in <io.h>) respectively. This
// allows working with files/directories whose paths are longer than
// MAX_PATH (260 chars).
//
// This file is only used on Windows, it's empty on other platforms.

#if defined(_WIN32)

#include <ctype.h>
#include <errno.h>
#include <wctype.h>

#include <google/protobuf/stubs/io_win32.h>

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace google {
namespace protobuf {
namespace stubs {
namespace {

using std::string;
using std::unique_ptr;
using std::wstring;

template <typename char_type>
struct CharTraits {
  static bool is_alpha(char_type ch);
};

template <>
struct CharTraits<char> {
  static bool is_alpha(char ch) { return isalpha(ch); }
};

template <>
struct CharTraits<wchar_t> {
  static bool is_alpha(wchar_t ch) { return iswalpha(ch); }
};

template <typename char_type>
bool has_drive_letter(const char_type* ch) {
  return CharTraits<char_type>::is_alpha(ch[0]) && ch[1] == ':';
}

template <typename char_type>
bool has_unc_prefix(const std::basic_string<char_type>& path) {
  return path.size() > 4 && path[0] == '\\' && path[1] == '\\' &&
         path[2] == '?' && path[3] == '\\';
}

template <typename char_type>
bool is_path_absolute(const std::basic_string<char_type>& path) {
  return (path.size() > 2 && path[1] == ':') || has_unc_prefix(path);
}

template <typename char_type>
bool is_separator(char_type c) {
  return c == '/' || c == '\\';
}

bool is_drive_relative(const string& s) {
  return s.size() >= 2 && isalpha(s[0]) && s[1] == ':' &&
         (s.size() == 2 || !is_separator(s[2]));
}

void replace_directory_separators(WCHAR* p) {
  for (; *p != L'\0'; ++p) {
    if (*p == L'/') {
      *p = L'\\';
    }
  }
}

wstring get_cwd() {
  DWORD result = ::GetCurrentDirectoryW(0, NULL);
  std::unique_ptr<WCHAR[]> cwd(new WCHAR[result]);
  ::GetCurrentDirectoryW(result, cwd.get());
  cwd.get()[result - 1] = 0;
  replace_directory_separators(cwd.get());
  return std::move(wstring(cwd.get()));
}

wstring join_paths(const wstring& path1, const wstring& path2) {
  if (path1.empty() || is_path_absolute(path2)) {
    return path2;
  }
  if (path2.empty()) {
    return path1;
  }

  if (is_separator(path1.back())) {
    return is_separator(path2.front()) ? (path1 + path2.substr(1))
                                       : (path1 + path2);
  } else {
    return is_separator(path2.front()) ? (path1 + path2)
                                       : (path1 + L'\\' + path2);
  }
}

string normalize(string path) {
  if (has_unc_prefix(path)) {
    path = path.substr(4);
  }

  static const string dot(".");
  static const string dotdot("..");

  std::vector<string> segments;
  int segment_start = -1;
  // Find the path segments in `path` (separated by "/").
  for (int i = 0;; ++i) {
    if (!is_separator(path[i]) && path[i] != '\0') {
      // The current character does not end a segment, so start one unless it's
      // already started.
      if (segment_start < 0) {
        segment_start = i;
      }
    } else if (segment_start >= 0 && i > segment_start) {
      // The current character is "/" or "\0", so this ends a segment.
      // Add that to `segments` if there's anything to add; handle "." and "..".
      string segment(path, segment_start, i - segment_start);
      segment_start = -1;
      if (segment == dotdot) {
        if (!segments.empty() && !has_drive_letter(segments[0].c_str())) {
          segments.pop_back();
        }
      } else if (segment != dot) {
        segments.push_back(segment);
      }
    }
    if (path[i] == '\0') {
      break;
    }
  }

  // Handle the case when `path` is just a drive specifier (or some degenerate
  // form of it, e.g. "c:\..").
  if (segments.size() == 1 && segments[0].size() == 2 &&
      has_drive_letter(segments[0].c_str())) {
    return segments[0] + '\\';
  }

  // Join all segments.
  bool first = true;
  std::ostringstream result;
  for (const auto& s : segments) {
    if (!first) {
      result << '\\';
    }
    first = false;
    result << s;
  }
  return result.str();
}

wstring as_wchar_path(const string& path) {
  int len =
      ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.size(), NULL, 0);
  std::unique_ptr<WCHAR[]> wbuf(new WCHAR[len + 1]);
  ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.size(), wbuf.get(),
                        len + 1);
  wbuf.get()[len] = 0;
  replace_directory_separators(wbuf.get());
  return std::move(wstring(wbuf.get()));
}

bool as_windows_path(const string& path, size_t max_path,
                            wstring* result) {
  if (path.empty()) {
    result->clear();
    return true;
  }
  if (is_separator(path[0]) || is_drive_relative(path)) {
    return false;
  }

  *result = as_wchar_path(normalize(path));
  if (!is_path_absolute(path)) {
    *result = join_paths(get_cwd(), *result);
  }
  if (result->size() >= max_path && !has_unc_prefix(*result)) {
    *result = wstring(L"\\\\?\\") + *result;
  }
  return true;
}

}  // namespace

int win32_open(const char* path, int flags, int mode) {
  wstring wpath;
  if (!as_windows_path(path, MAX_PATH, &wpath)) {
    errno = ENOENT;
    return -1;
  }
  return ::_wopen(wpath.c_str(), flags, mode);
}

int win32_mkdir(const char* path, int _mode) {
  // CreateDirectoryA's limit is 248 chars, see MSDN.
  // https://msdn.microsoft.com/en-us/library/windows/desktop/aa363855(v=vs.85).aspx
  // This limit presumably includes the null-terminator, because other
  // functions that have the MAX_PATH limit, such as CreateFileA,
  // actually include it.
  wstring wpath;
  if (!as_windows_path(path, 248, &wpath)) {
    errno = ENOENT;
    return -1;
  }
  return ::_wmkdir(wpath.c_str());
}

int win32_access(const char* path, int mode) {
  wstring wpath;
  if (!as_windows_path(path, MAX_PATH, &wpath)) {
    errno = ENOENT;
    return -1;
  }
  return ::_waccess(wpath.c_str(), mode);
}

wstring testonly_path_to_winpath(const string& path, size_t max_path) {
  wstring wpath;
  as_windows_path(path, max_path, &wpath);
  return wpath;
}

}  // namespace stubs
}  // namespace protobuf
}  // namespace google

#endif  // defined(_WIN32)

