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
#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_OPERATIONS_H_
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_OPERATIONS_H_

#include <windows.h>

#include <memory>

namespace windows_util {

using std::unique_ptr;

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
enum {
  IS_JUNCTION_YES = 0,
  IS_JUNCTION_NO = 1,
  IS_JUNCTION_ERROR = 2,
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
int IsJunctionOrDirectorySymlink(const WCHAR* path);

// Computes the long version of `path` if it has any 8dot3 style components.
// Returns true upon success and sets `result` to point to the buffer.
// `path` must be an absolute, normalized, Windows style path, with a "\\?\"
// prefix if its longer than MAX_PATH. The result will have a "\\?\" prefix if
// and only if `path` had one as well. (It's the caller's responsibility to keep
// or remove this prefix.)
bool GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result);

}  // namespace windows_util

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_FILE_OPERATIONS_H_
