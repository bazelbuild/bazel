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
// allows working with files/directories whose paths is longer than
// MAX_PATH (260 chars).
//
// This file is only used on Windows, it's empty on other platforms.

#if defined(_WIN32)

#include <Windows.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <io.h>

#include <string>
#include <memory>

#include <google/protobuf/io/io_win32.h>

namespace google {
namespace protobuf {
namespace io {
namespace win32 {

template <typename char_type>
static bool has_unc_prefix(const std::basic_string<char_type>& path) {
  return path.size() > 4 && path[0] == '\\' && path[1] == '\\' &&
         path[2] == '?' && path[3] == '\\';
}

template <typename char_type>
static bool is_path_absolute(const std::basic_string<char_type>& path) {
  return (path.size() > 2 && path[1] == ':') || has_unc_prefix(path);
}

template <typename char_type>
static bool is_separator(char_type c) {
  return c == '/' || c == '\\';
}

static void replace_directory_separators(WCHAR* p) {
  for (; *p != L'\0'; ++p) {
    if (*p == L'/') {
      *p = L'\\';
    }
  }
}

static std::wstring get_cwd() {
  DWORD result = ::GetCurrentDirectoryW(0, NULL);
  std::unique_ptr<WCHAR[]> cwd(new WCHAR[result]);
  ::GetCurrentDirectoryW(result, cwd.get());
  cwd.get()[result - 1] = 0;
  replace_directory_separators(cwd.get());
  return std::move(std::wstring(cwd.get()));
}

static std::wstring join_paths(const std::wstring& path1, 
                               const std::wstring& path2) {
  if (path1.empty() || is_path_absolute(path2)) {
    return path2;
  }
  if (path2.empty()) {
    return path1;
  }

  if (is_separator(path1.back())) {
    return is_separator(path2.front())
        ? (path1 + path2.substr(1))
        : (path1 + path2);
  } else {
    return is_separator(path2.front())
        ? (path1 + path2)
        : (path1 + L'\\' + path2);
  }
}

static std::wstring as_wchar_path(const std::string& path) {
  int len = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.size(),
                                  NULL, 0);
  std::unique_ptr<WCHAR[]> wbuf(new WCHAR[len + 1]);
  ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.size(),
                        wbuf.get(), len + 1);
  wbuf.get()[len] = 0;
  replace_directory_separators(wbuf.get());
  return std::move(std::wstring(wbuf.get()));
}

static std::wstring as_windows_path(const std::string& path,
                                    size_t max_path) {
  std::wstring wpath(as_wchar_path(path));
  if (!is_path_absolute(path)) {
    wpath = join_paths(get_cwd(), wpath);
  }
  if (wpath.size() >= max_path && !has_unc_prefix(wpath)) {
    wpath = std::wstring(L"\\\\?\\") + wpath;
  }
  return wpath;
}

int open(const char* path, int flags, int mode) {
  return ::_wopen(as_windows_path(path, MAX_PATH).c_str(), flags, mode);
}

int mkdir(const char* name, int _mode) {
  // CreateDirectoryA's limit is 248 chars, see MSDN.
  // https://msdn.microsoft.com/en-us/library/windows/desktop/aa363855(v=vs.85).aspx
  // This limit presumably includes the null-terminator, because other
  // functions that have the MAX_PATH limit, such as CreateFileA,
  // actually include it.
  return ::_wmkdir(as_windows_path(name, 248).c_str());
}

int access(const char* pathname, int mode) {
  return ::_waccess(as_windows_path(pathname, MAX_PATH).c_str(), mode);
}

}  // namespace win32
}  // namespace io
}  // namespace protobuf
}  // namespace google

#endif  // defined(_WIN32)
