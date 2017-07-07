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

#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__

#include <windows.h>

#include <functional>
#include <memory>
#include <string>

namespace bazel {
namespace windows {

using std::function;
using std::string;
using std::unique_ptr;
using std::wstring;

// A wrapper for the `HANDLE` type that calls CloseHandle in its d'tor.
// WARNING: do not use for HANDLE returned by FindFirstFile; those must be
// closed with FindClose (otherwise they aren't closed properly).
struct AutoHandle {
  AutoHandle(HANDLE _handle = INVALID_HANDLE_VALUE) : handle(_handle) {}

  ~AutoHandle() {
    ::CloseHandle(handle);  // succeeds if handle == INVALID_HANDLE_VALUE
    handle = INVALID_HANDLE_VALUE;
  }

  bool IsValid() { return handle != INVALID_HANDLE_VALUE && handle != NULL; }

  AutoHandle& operator=(const HANDLE& rhs) {
    ::CloseHandle(handle);
    handle = rhs;
    return *this;
  }

  operator HANDLE() const { return handle; }

  HANDLE handle;
};

string GetLastErrorString(const string& cause);

// Same as `AsExecutablePathForCreateProcess` except it won't quote the result.
string AsShortPath(string path, function<wstring()> path_as_wstring,
                   string* result);

// Computes a path suitable as the executable part in CreateProcessA's cmdline.
//
// The null-terminated executable path for CreateProcessA has to fit into
// MAX_PATH, therefore the limit for the executable's path is MAX_PATH - 1
// (not including null terminator). This method attempts to convert the input
// `path` to a short format to fit it into the MAX_PATH - 1 limit.
//
// `path` must be either an absolute, normalized, Windows-style path with drive
// letter (e.g. "c:\foo\bar.exe", but no "\foo\bar.exe"), or must be just a file
// name (e.g. "cmd.exe") that's shorter than MAX_PATH (without null-terminator).
// In both cases, `path` must be unquoted.
//
// `path_as_wstring` must be a function that retrieves `path` as (or converts it
// to) a wstring, without performing any transformations on the path.
//
// If this function succeeds, it returns an empty string (indicating no error),
// and sets `result` to the resulting path, which is always quoted, and is
// always at most MAX_PATH + 1 long (MAX_PATH - 1 without null terminator, plus
// two quotes). If there's any error, this function returns the error message.
//
// If `path` is at most MAX_PATH - 1 long (not including null terminator), the
// result will be that (plus quotes).
// Otherwise this method attempts to compute an 8dot3 style short name for
// `path`, and if that succeeds and the result is at most MAX_PATH - 1 long (not
// including null terminator), then that will be the result (plus quotes).
// Otherwise this function fails and returns an error message.
string AsExecutablePathForCreateProcess(const string& path,
                                        function<wstring()> path_as_wstring,
                                        string* result);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__
