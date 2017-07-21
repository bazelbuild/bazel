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
__declspec(noreturn) void die(const char* format, ...) PRINTF_ATTRIBUTE(1, 2);

// Prints the specified error message.
void PrintError(const char* format, ...) PRINTF_ATTRIBUTE(1, 2);

// Strip the .exe extension from binary path.
//
// On Windows, if the binary path is foo/bar/bin.exe then return foo/bar/bin
std::string GetBinaryPathWithoutExtension(const std::string& binary);

// Add exectuable extension to binary path
//
// On Windows, if the binary path is foo/bar/bin then return foo/bar/bin.exe
std::string GetBinaryPathWithExtension(const std::string& binary);

// Escape a command line argument.
//
// If the argument has space, then we quote it.
// Escape \ to \\
// Escape " to \"
std::string GetEscapedArgument(const std::string& argument);

// Check if a file exists at a given path.
bool DoesFilePathExist(const std::string& path);

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_UTIL_LAUNCHER_UTIL_H_
