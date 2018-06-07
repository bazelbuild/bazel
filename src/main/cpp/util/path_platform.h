// Copyright 2018 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_CPP_UTIL_PATH_PLATFORM_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_PATH_PLATFORM_H_

#include <string>

namespace blaze_util {

// Convert a path from Bazel internal form to underlying OS form.
// On Unixes this is an identity operation.
// On Windows, Bazel internal form is cygwin path, and underlying OS form
// is Windows path.
std::string ConvertPath(const std::string &path);

// Converts `path` to a string that's safe to pass as path in a JVM flag.
// See https://github.com/bazelbuild/bazel/issues/2576
std::string PathAsJvmFlag(const std::string &path);

// Compares two absolute paths. Necessary because the same path can have
// multiple different names under msys2: "C:\foo\bar" or "C:/foo/bar"
// (Windows-style) and "/c/foo/bar" (msys2 style). Returns if the paths are
// equal.
bool CompareAbsolutePaths(const std::string &a, const std::string &b);

// Split a path to dirname and basename parts.
std::pair<std::string, std::string> SplitPath(const std::string &path);

bool IsDevNull(const char *path);

// Returns true if `path` is the root directory or a Windows drive root.
bool IsRootDirectory(const std::string &path);

// Returns true if `path` is absolute.
bool IsAbsolute(const std::string &path);

// Returns the given path in absolute form.  Does not change paths that are
// already absolute.
//
// If called from working directory "/bar":
//   MakeAbsolute("foo") --> "/bar/foo"
//   MakeAbsolute("/foo") ---> "/foo"
//   MakeAbsolute("C:/foo") ---> "C:/foo"
std::string MakeAbsolute(const std::string &path);

// TODO(bazel-team) consider changing the path(_platform) header split to be a
// path.h and path_windows.h split, which would make it clearer what functions
// are included by an import statement. The downside to this gain in clarity
// is that this would add more complexity to the implementation file(s)? of
// path.h, which would have to have the platform-specific implementations.
#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
const wchar_t *RemoveUncPrefixMaybe(const wchar_t *ptr);
void AddUncPrefixMaybe(std::wstring *path);

std::pair<std::wstring, std::wstring> SplitPathW(const std::wstring &path);

bool IsRootDirectoryW(const std::wstring &path);

bool AsWindowsPath(const std::string &path, std::string *result,
                   std::string *error);

// Returns a normalized form of the input `path`.
//
// `path` must be a relative or absolute Windows path, it may use "/" instead of
// "\" but must not be a Unix-style (MSYS) path.
// The result won't have a UNC prefix, even if `path` did.
//
// Normalization means removing "." references, resolving ".." references, and
// deduplicating "/" characters while converting them to "\".
// For example if `path` is "foo/../bar/.//qux", the result is "bar\qux".
//
// Uplevel references that cannot go any higher in the directory tree are simply
// ignored, e.g. "c:/.." is normalized to "c:\" and "../../foo" is normalized to
// "foo".
//
// Visible for testing, would be static otherwise.
std::string NormalizeWindowsPath(std::string path);

// Converts a UTF8-encoded `path` to a normalized, widechar Windows path.
//
// Returns true if conversion succeeded and sets the contents of `result` to it.
//
// The input `path` may be an absolute or relative Windows path.
//
// The returned path is normalized (see NormalizeWindowsPath).
//
// If `path` had a "\\?\" prefix then the function assumes it's already Windows
// style and converts it to wstring without any alterations.
// Otherwise `path` is normalized and converted to a Windows path and the result
// won't have a "\\?\" prefix even if it's longer than MAX_PATH (adding the
// prefix is the caller's responsibility).
//
// The method recognizes current-drive-relative Windows paths ("\foo") turning
// them into absolute paths ("c:\foo").
bool AsWindowsPath(const std::string &path, std::wstring *result,
                   std::string *error);

bool AsAbsoluteWindowsPath(const std::string &path, std::wstring *wpath,
                           std::string *error);

// Same as `AsWindowsPath`, but returns a lowercase 8dot3 style shortened path.
// Result will never have a UNC prefix, nor a trailing "/" or "\".
// Works also for non-existent paths; shortens as much of them as it can.
// Also works for non-existent drives.
bool AsShortWindowsPath(const std::string &path, std::string *result,
                        std::string *error);

template <typename char_type>
bool IsPathSeparator(char_type ch);

template <typename char_type>
bool HasDriveSpecifierPrefix(const char_type *ch);

#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_PATH_PLATFORM_H_
