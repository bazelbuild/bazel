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

// Returns the given path in absolute form, taking into account a
// possible starting environment variable, so that we can accept
// standard path variables like %USERPROFILE% or ${BAZEL}. For
// simplicity, we implement only those two forms, not $BAZEL.
//
//   MakeAbsolute("foo") in wd "/bar" --> "/bar/foo"
//   MakeAbsoluteAndResolveEnvvars("%USERPROFILE%/foo") -->
//       "C:\Users\bazel-user\foo"
//   MakeAbsoluteAndResolveEnvvars("${BAZEL}/foo") --> "/opt/bazel/foo"
std::string MakeAbsoluteAndResolveEnvvars(const std::string &path);

// TODO(bazel-team) consider changing the path(_platform) header split to be a
// path.h and path_windows.h split, which would make it clearer what functions
// are included by an import statement. The downside to this gain in clarity
// is that this would add more complexity to the implementation file(s)? of
// path.h, which would have to have the platform-specific implementations.
#if defined(_WIN32) || defined(__CYGWIN__)
bool IsDevNull(const wchar_t *path);

bool IsAbsolute(const std::wstring &path);

const wchar_t *RemoveUncPrefixMaybe(const wchar_t *ptr);

void AddUncPrefixMaybe(std::wstring *path);

std::pair<std::wstring, std::wstring> SplitPathW(const std::wstring &path);

bool IsRootDirectoryW(const std::wstring &path);

bool TestOnly_NormalizeWindowsPath(const std::string &path,
                                   std::string *result);

// Converts 'path' to Windows style.
//
// 'path' is absolute or relative or current-drive-relative (e.g.
// "\foo"), possibly non-normalized, possibly using slash as separator. If it
// starts with the UNC prefix, the function won't process it further, just
// copies it to 'result'.
//
// 'result' equals 'path' if 'path' started with the UNC prefix, otherwise
// 'result' is normalized, using backslash as separator.
//
// Encoding: there is no assumption about encoding, 'path' is read as ASCII
// (Latin-1) and 'result' uses the same encoding.
bool AsWindowsPath(const std::string &path, std::string *result,
                   std::string *error);

// Converts 'path' to Windows style.
//
// Same as the other AsWindowsPath methods, but 'path' is encoded as multibyte
// and 'result' is widechar. (MSDN does not clarify what multibyte means. The
// function uses blaze_util::WstringToCstring.)
bool AsWindowsPath(const std::string &path, std::wstring *result,
                   std::string *error);

// Converts 'path' to Windows style.
//
// Same as the other AsWindowsPath methods, but 'path' and 'result' are
// widechar.
bool AsWindowsPath(const std::wstring &path, std::wstring *result,
                   std::string *error);

// Converts 'path' to absolute, Windows-style path.
//
// Same as AsWindowsPath, but 'result' is always absolute and always has a UNC
// prefix.
bool AsAbsoluteWindowsPath(const std::string &path, std::wstring *result,
                           std::string *error);

// Converts 'path' to absolute, Windows-style path.
//
// Same as AsWindowsPath, but 'result' is always absolute and always has a UNC
// prefix.
bool AsAbsoluteWindowsPath(const std::wstring &path, std::wstring *result,
                           std::string *error);

// Converts 'path' to absolute, shortened, Windows-style path.
//
// Same as `AsWindowsPath`, but 'result' is always absolute, lowercase,
// 8dot3-style shortened path, without trailing backslash and without UNC
// prefix.
//
// Works even for non-existent paths (and non-existent drives), shortening the
// existing segments and leaving the rest unshortened.
bool AsShortWindowsPath(const std::string &path, std::string *result,
                        std::string *error);

#endif  // defined(_WIN32) || defined(__CYGWIN__)
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_PATH_PLATFORM_H_
