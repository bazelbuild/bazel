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

#ifndef BAZEL_SRC_TOOLS_LAUNCHER_UTIL_LAUNCHER_UTIL_H_
#define BAZEL_SRC_TOOLS_LAUNCHER_UTIL_LAUNCHER_UTIL_H_

#define PRINTF_ATTRIBUTE(string_index, first_to_check)

#include <string>

namespace bazel {
namespace launcher {

std::string GetLastErrorString();

// Prints the specified error message and exits nonzero.
__declspec(noreturn) void die(const wchar_t* format, ...)
    PRINTF_ATTRIBUTE(1, 2);

// Prints the specified error message.
void PrintError(const wchar_t* format, ...) PRINTF_ATTRIBUTE(1, 2);

// Converts the specified path (Windows 8dot3 style short path) to its long form
//
// eg. C:\FO~1\BAR\B~1 -> C:\Foooo\Bar\bin.exe
// Note that: the given path must be an existing path.
std::wstring GetWindowsLongPath(const std::wstring& path);

// Strip the .exe extension from binary path.
//
// On Windows, if the binary path is foo/bar/bin.exe then return foo/bar/bin
std::wstring GetBinaryPathWithoutExtension(const std::wstring& binary);

// Add executable extension to binary path
//
// On Windows, if the binary path is foo/bar/bin then return foo/bar/bin.exe
std::wstring GetBinaryPathWithExtension(const std::wstring& binary);

// Escape a command line argument using Bash escaping syntax.
//
// If the argument has space, then we quote it. We escape quote with a backslash
// (from " to \") and escape a single backslash with another backslash (from \
// to \\).
std::wstring BashEscapeArg(const std::wstring& arg);

// Convert a path to an absolute Windows path with \\?\ prefix.
// This method will print an error and exit if it cannot convert the path.
std::wstring AsAbsoluteWindowsPath(const wchar_t* path);

// Check if a file exists at a given path.
bool DoesFilePathExist(const wchar_t* path);

// Check if a directory exists at a given path.
bool DoesDirectoryPathExist(const wchar_t* path);

// Delete a file at a given path.
bool DeleteFileByPath(const wchar_t* path);

// Delete a directory at a given path,.
// If it's a real directory, it must be empty
// If it's a junction, the target directory it points to doesn't have to be
// empty, the junction will be deleted regardless of the state of the target.
bool DeleteDirectoryByPath(const wchar_t* path);

// Get the value of a specific environment variable
//
// Return true if succeeded and the result is stored in buffer.
// Return false if the environment variable doesn't exist or the value is empty.
bool GetEnv(const std::wstring& env_name, std::wstring* buffer);

// Set the value of a specific environment variable
//
// Return true if succeeded, otherwise false.
bool SetEnv(const std::wstring& env_name, const std::wstring& value);

// Return a random string with a given length.
// The string consists of a-zA-Z0-9
std::wstring GetRandomStr(size_t len);

// Normalize a path to a Windows path in lower case
bool NormalizePath(const std::wstring& path, std::wstring* result);

// Get the base name from a normalized absoulute path
std::wstring GetBaseNameFromPath(const std::wstring& path);

// Get parent directory from a normalized absoulute path
std::wstring GetParentDirFromPath(const std::wstring& path);

// Calculate a relative path from `path` to `base`.
// This function expects normalized Windows path in lower case.
// `path` and `base` should be both absolute or both relative.
bool RelativeTo(const std::wstring& path, const std::wstring& base,
                std::wstring* result);

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_UTIL_LAUNCHER_UTIL_H_
