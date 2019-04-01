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
#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_H_
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_H_

#include <windows.h>

#include <memory>
#include <string>

namespace bazel {
namespace windows {

using std::unique_ptr;
using std::wstring;

template <typename char_type>
bool HasUncPrefix(const char_type* path) {
  // Return true iff `path` starts with "\\?\", "\\.\", or "\??\".
  return path[0] == '\\' &&
         ((path[1] == '\\' && (path[2] == '?' || path[2] == '.')) ||
          (path[1] == '?' && path[2] == '?')) &&
         path[3] == '\\';
}

template <typename char_type>
bool IsDevNull(const char_type* path) {
  return (path[0] == 'N' || path[0] == 'n') &&
         (path[1] == 'U' || path[1] == 'u') &&
         (path[2] == 'L' || path[2] == 'l');
}

std::wstring AddUncPrefixMaybe(const std::wstring& path);

std::wstring RemoveUncPrefixMaybe(const std::wstring& path);

bool IsAbsoluteNormalizedWindowsPath(const std::wstring& p);

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
enum {
  IS_JUNCTION_YES = 0,
  IS_JUNCTION_NO = 1,
  IS_JUNCTION_ERROR = 2,
};

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
struct DeletePathResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kDoesNotExist = 2,
    kDirectoryNotEmpty = 3,
    kAccessDenied = 4,
  };
};

struct CreateJunctionResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kTargetNameTooLong = 2,
    kAlreadyExistsWithDifferentTarget = 3,
    kAlreadyExistsButNotJunction = 4,
    kAccessDenied = 5,
    kDisappeared = 6,
  };
};

struct ReadJunctionResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kDoesNotExist = 2,
    kAccessDenied = 3,
    kNotAJunction = 4,
  };
};

// Determines whether `path` is a junction (or directory symlink).
//
// `path` should be an absolute, normalized, Windows-style path, with "\\?\"
// prefix if it's longer than MAX_PATH.
//
// To read about differences between junctions and directory symlinks,
// see http://superuser.com/a/343079. In Bazel we only ever create junctions.
//
// Returns:
// - IS_JUNCTION_YES, if `path` exists and is either a directory junction or a
//   directory symlink
// - IS_JUNCTION_NO, if `path` exists but is neither a directory junction nor a
//   directory symlink; also when `path` is a symlink to a directory but it was
//   created using "mklink" instead of "mklink /d", as such symlinks don't
//   behave the same way as directories (e.g. they can't be listed)
// - IS_JUNCTION_ERROR, if `path` doesn't exist or some error occurred
int IsJunctionOrDirectorySymlink(const WCHAR* path, std::wstring* error);

// Computes the long version of `path` if it has any 8dot3 style components.
// Returns the empty string upon success, or a human-readable error message upon
// failure.
// `path` must be an absolute, normalized, Windows style path, with a "\\?\"
// prefix if it's longer than MAX_PATH. The result will have a "\\?\" prefix if
// and only if `path` had one as well. (It's the caller's responsibility to keep
// or remove this prefix.)
// TODO(laszlocsomor): update GetLongPath so it succeeds even if the path does
// not (fully) exist.
wstring GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result);

// Creates a junction at `name`, pointing to `target`.
// Returns CreateJunctionResult::kSuccess if it could create the junction, or if
// the junction already exists with the same target.
// If the junction's name already exists as an empty directory, this function
// will turn it into a junction and return kSuccess.
// Otherwise returns one of the other CreateJunctionResult::k* constants for
// known error cases, or CreateJunctionResult::kError for unknown error cases.
// When the function returns CreateJunctionResult::kError, and `error` is
// non-null, the function writes an error message into `error`. If the return
// value is anything other than CreateJunctionResult::kError, then this function
// ignores the  `error` argument.
//
// Neither `junction_name` nor `junction_target` needs to have a "\\?\" prefix,
// not even if they are longer than MAX_PATH, though it's okay if they do. This
// function will add the right prefixes as necessary.
int CreateJunction(const wstring& junction_name, const wstring& junction_target,
                   wstring* error);

int ReadJunction(const wstring& path, wstring* result, wstring* error);

// Deletes the file, junction, or empty directory at `path`.
// Returns DELETE_PATH_SUCCESS if it successfully deleted the path, otherwise
// returns one of the other DELETE_PATH_* constants (e.g. when the directory is
// not empty or the file is in use by another process).
// Returns DELETE_PATH_ERROR for unexpected errors. If `error` is not null, the
// function writes an error message into it.
int DeletePath(const wstring& path, wstring* error);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_H_
