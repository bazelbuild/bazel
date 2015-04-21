// Copyright 2014 Google Inc. All rights reserved.
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

#ifndef DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_H__
#define DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_H__

#include <pwd.h>
#include <stdarg.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <string>
#include <vector>

#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/port.h"

namespace blaze {

using std::string;

string GetUserName();

// Return the path to the JVM launcher.
string GetJvm();

// Returns the given path in absolute form.  Does not change paths that are
// already absolute.
//
// If called from working directory "/bar":
//   MakeAbsolute("foo") --> "/bar/foo"
//   MakeAbsolute("/foo") ---> "/foo"
string MakeAbsolute(string path);

// mkdir -p path. All newly created directories use the given mode.
// Returns -1 on failure, sets errno.
int MakeDirectories(string path, int mode);

// Replaces 'content' with contents of file 'filename'.
// Returns false on error.
bool ReadFile(const string &filename, string *content);

// Replaces 'content' with contents of file descriptor 'fd'.
// Returns false on error.
bool ReadFileDescriptor(int fd, string *content);

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const string &content, const string &filename);

// Returns true iff the current terminal can support color and cursor movement.
bool IsStandardTerminal();

// Returns the number of columns of the terminal to which stdout is
// connected, or 80 if there is no such terminal.
int GetTerminalColumns();

// blaze's JVM arch is set at build time (--java_cpu), since the blaze java
// process includes native code.
bool Is64BitBlazeJavabase();

// Adds JVM arguments particular to running blaze with JVM v3 or higher.
void AddJVMSpecificArguments(const string &host_javabase,
                             std::vector<string> *result);

void ExecuteProgram(string exe, const std::vector<string>& args_vector);

void ReExecute(const string &executable, int argc, const char *argv[]);

// If 'arg' matches 'key=value', returns address of 'value'.
// If it matches 'key' alone, returns address of next_arg.
// Returns NULL otherwise.
const char* GetUnaryOption(const char *arg, const char *next_arg,
                                  const char *key);

// Returns true iff 'arg' equals 'key'.
// Dies with a syntax error if arg starts with 'key='.
// Returns NULL otherwise.
bool GetNullaryOption(const char *arg, const char *key);

bool CheckValidPort(const string &str, const string &option, string *error);

// Enable messages mostly of interest to developers.
bool VerboseLogging();

// Read the JVM version from a file descriptor. The fd should point
// to the output of a "java -version" execution and is supposed to contains
// a string of the form 'version "version-number"' in the first 255 bytes.
// If the string is found, version-number is returned, else the empty string
// is returned.
string ReadJvmVersion(int fd);

// Get the version string from the given java executable. The java executable
// is supposed to output a string in the form '.*version ".*".*'. This method
// will return the part in between the two quote or the empty string on failure
// to match the good string.
string GetJvmVersion(string java_exe);

// Returns true iff jvm_version is at least the version specified by
// version_spec.
// jvm_version is supposed to be a string specifying a java runtime version
// as specified by the JSR-56 appendix A. version_spec is supposed to be a
// version is the format [0-9]+(.[1-9]+)*.
bool CheckJavaVersionIsAtLeast(string jvm_version, string version_spec);

}  // namespace blaze
#endif  // DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_H__
