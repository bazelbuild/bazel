// Copyright 2014 The Bazel Authors. All rights reserved.
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
//
// blaze_util.h: Miscellaneous utility functions used by the blaze.cc
//               Blaze client.
//

#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_H_

#include <sys/types.h>

#include <sstream>
#include <string>
#include <vector>

namespace blaze {

extern const char kServerPidFile[];

// TODO(laszlocsomor) 2016-11-21: remove kServerPidSymlink after 2017-05-01
// (~half a year from writing this comment). By that time old Bazel clients that
// used to write PID symlinks will probably no longer be in use.
extern const char kServerPidSymlink[];

std::string GetUserName();

// Returns the given path in absolute form.  Does not change paths that are
// already absolute.
//
// If called from working directory "/bar":
//   MakeAbsolute("foo") --> "/bar/foo"
//   MakeAbsolute("/foo") ---> "/foo"
//   MakeAbsolute("C:/foo") ---> "C:/foo"
std::string MakeAbsolute(const std::string &path);

// Replaces 'content' with contents of file 'filename'.
// If `max_size` is positive, the method reads at most that many bytes;
// otherwise the method reads the whole file.
// Returns false on error. Can be called from a signal handler.
bool ReadFile(const std::string &filename, std::string *content,
              int max_size = 0);

// Replaces 'content' with contents of file descriptor 'fd'.
// If `max_size` is positive, the method reads at most that many bytes; if it
// otherwise the method reads the whole file.
// Returns false on error. Can be called from a signal handler.
bool ReadFileDescriptor(int fd, std::string *content, int max_size = 0);

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const std::string &content, const std::string &filename);

// Writes `size` bytes from `data` into file 'filename' and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const void* data, size_t size, const std::string &filename);

// Unlinks the file given by 'file_path'.
// Returns true on success. In case of failure sets errno.
bool UnlinkPath(const std::string &file_path);

// Returns true iff the current terminal is running inside an Emacs.
bool IsEmacsTerminal();

// Returns true iff the current terminal can support color and cursor movement.
bool IsStandardTerminal();

// Returns the number of columns of the terminal to which stdout is
// connected, or 80 if there is no such terminal.
int GetTerminalColumns();

// If 'arg' matches 'key=value', returns address of 'value'.
// If it matches 'key' alone, returns address of next_arg.
// Returns NULL otherwise.
const char* GetUnaryOption(const char *arg,
                           const char *next_arg,
                           const char *key);

// Returns true iff 'arg' equals 'key'.
// Dies with a syntax error if arg starts with 'key='.
// Returns false otherwise.
bool GetNullaryOption(const char *arg, const char *key);

// Enable messages mostly of interest to developers.
bool VerboseLogging();

// Read the JVM version from a string. The string should contain the output of a
// "java -version" execution and is supposed to contain a string of the form
// 'version "version-number"' in the first 255 bytes. If the string is found,
// version-number is returned, else the empty string is returned.
std::string ReadJvmVersion(const std::string &version_string);

// Get the version string from the given java executable. The java executable
// is supposed to output a string in the form '.*version ".*".*'. This method
// will return the part in between the two quote or the empty string on failure
// to match the good string.
std::string GetJvmVersion(const std::string &java_exe);

// Returns true iff jvm_version is at least the version specified by
// version_spec.
// jvm_version is supposed to be a string specifying a java runtime version
// as specified by the JSR-56 appendix A. version_spec is supposed to be a
// version is the format [0-9]+(.[1-9]+)*.
bool CheckJavaVersionIsAtLeast(const std::string &jvm_version,
                               const std::string &version_spec);

// Returns true iff arg is a valid command line argument for bazel.
bool IsArg(const std::string& arg);

// Returns the string representation of `value`.
// Workaround for mingw where std::to_string is not implemented.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52015.
template <typename T>
std::string ToString(const T &value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_H_
