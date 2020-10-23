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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
#define SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE 0x2
#endif

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

template <typename char_type>
bool HasDriveSpecifierPrefix(const char_type* p) {
  if (HasUncPrefix(p)) {
    return iswalpha(p[4]) && p[5] == ':' && (p[6] == '\\' || p[6] == '/');
  } else {
    return iswalpha(p[0]) && p[1] == ':' && (p[2] == '\\' || p[2] == '/');
  }
}

std::wstring AddUncPrefixMaybe(const std::wstring& path);

std::wstring RemoveUncPrefixMaybe(const std::wstring& path);

bool IsAbsoluteNormalizedWindowsPath(const std::wstring& p);

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
struct IsSymlinkOrJunctionResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kDoesNotExist = 2,
  };
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

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
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

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
struct CreateSymlinkResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kTargetIsDirectory = 2,
  };
};

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
struct ReadSymlinkOrJunctionResult {
  enum {
    kSuccess = 0,
    kError = 1,
    kAccessDenied = 2,
    kDoesNotExist = 3,
    kNotALink = 4,
    kUnknownLinkType = 5,
  };
};

// Determines whether `path` is a junction (or directory symlink).
//
// `path` should be an absolute, normalized, Windows-style path, with "\\?\"
// prefix if it's longer than MAX_PATH.
//
// To read about differences between junctions and directory symlinks,
// see http://superuser.com/a/343079. In Bazel we only ever create junctions.
int IsSymlinkOrJunction(const WCHAR* path, bool* result, wstring* error);

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

// Creates a symlink at `symlink_name`, pointing to `symlink_target`.
// Returns CreateSymlinkResult::kSuccess if could create the symlink.
// If the target is a directory, this function will return
// CreateSymlinkResult::kTargetIsDirectory since a junction is preferred
// instead. When the function returns CreateSymlinkResult::kError and `error`
// is non-null then `error` receives an error message.
int CreateSymlink(const wstring& symlink_name, const wstring& symlink_target,
                  wstring* error);

// Reads the symlink or junction into 'result'.
// Returns a value from 'ReadSymlinkOrJunctionResult'.
// When the method returns 'ReadSymlinkOrJunctionResult::kError' and 'error' is
// non-null then 'error' receives an error message.
int ReadSymlinkOrJunction(const wstring& path, wstring* result, wstring* error);

// Deletes the file, junction, or empty directory at `path`.
// Returns DELETE_PATH_SUCCESS if it successfully deleted the path, otherwise
// returns one of the other DELETE_PATH_* constants (e.g. when the directory is
// not empty or the file is in use by another process).
// Returns DELETE_PATH_ERROR for unexpected errors. If `error` is not null, the
// function writes an error message into it.
int DeletePath(const wstring& path, wstring* error);

// Returns a normalized form of the input `path`.
//
// Normalization:
//   Normalization means removing "." references, resolving ".." references,
//   and deduplicating "/" characters while converting them to "\\".  For
//   example if `path` is "foo/../bar/.//qux", the result is "bar\\qux".
//
//   Uplevel references ("..") that cannot go any higher in the directory tree
//   are preserved if the path is relative, and ignored if the path is
//   absolute, e.g. "../../foo" is normalized to "..\\..\\foo" but "c:/.." is
//   normalized to "c:\\".
//
//   This method does not check the semantics of the `path` beyond checking if
//   it starts with a directory separator. Illegal paths such as one with a
//   drive specifier in the middle (e.g. "foo/c:/bar") are accepted -- it's the
//   caller's responsibility to pass a path that, when normalized, will be
//   semantically correct.
//
//   Current directory references (".") are preserved if and only if they are
//   the only path segment, so "./" becomes "." but "./foo" becomes "foo".
//
// Arguments:
//   `path` must be a relative or absolute Windows path, it may use "/" instead
//   of "\\". The path should not start with "/" or "\\".
//
// Result:
//   Returns false if and only if the path starts with a directory separator.
//
//   The result won't have a UNC prefix, even if `path` did. The result won't
//   have a trailing "\\" except when and only when the path is normalized to
//   just a drive specifier (e.g. when `path` is "c:/" or "c:/foo/.."). The
//   result will preserve the casing of the input, so "D:/Bar" becomes
//   "D:\\Bar".
std::string Normalize(const std::string& p);
std::wstring Normalize(const std::wstring& p);

bool GetCwd(std::wstring* result, DWORD* err_code);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_H_
