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

#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_PLATFORM_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_PLATFORM_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "src/main/cpp/util/port.h"

namespace blaze {

struct GlobalVariables;

class SignalHandler {
 public:
  typedef void (* Callback)();

  static SignalHandler& Get() { return INSTANCE; }
  GlobalVariables* GetGlobals() { return _globals; }
  void CancelServer() { _cancel_server(); }
  void Install(GlobalVariables* globals, Callback cancel_server);
  ATTRIBUTE_NORETURN void PropagateSignalOrExit(int exit_code);

 private:
  static SignalHandler INSTANCE;

  GlobalVariables* _globals;
  Callback _cancel_server;

  SignalHandler() : _globals(nullptr), _cancel_server(nullptr) {}
};

// A signal-safe version of fprintf(stderr, ...).
void SigPrintf(const char *format, ...);

std::string GetProcessIdAsString();

// Get the absolute path to the binary being executed.
std::string GetSelfPath();

// Returns the directory Bazel can use to store output.
std::string GetOutputRoot();

// Returns the current user's home directory, or the empty string if unknown.
// On Linux/macOS, this is $HOME. On Windows this is %USERPROFILE%.
std::string GetHomeDir();

// Returns the location of the global bazelrc file if it exists, otherwise "".
std::string FindSystemWideBlazerc();

// Warn about dubious filesystem types, such as NFS, case-insensitive (?).
void WarnFilesystemType(const std::string& output_base);

// Returns elapsed milliseconds since some unspecified start of time.
// The results are monotonic, i.e. subsequent calls to this method never return
// a value less than a previous result.
uint64_t GetMillisecondsMonotonic();

// Returns elapsed milliseconds since the process started.
uint64_t GetMillisecondsSinceProcessStart();

// Set cpu and IO scheduling properties. Note that this can take ~50ms
// on Linux, so it should only be called when necessary.
void SetScheduling(bool batch_cpu_scheduling, int io_nice_level);

// Returns the cwd for a process.
std::string GetProcessCWD(int pid);

bool IsSharedLibrary(const std::string& filename);

// Return the default path to the JDK used to run Blaze itself
// (must be an absolute directory).
std::string GetDefaultHostJavabase();

// Return the path to the JVM binary relative to a javabase, e.g. "bin/java".
std::string GetJavaBinaryUnderJavabase();

// Replace the current process with the given program in the current working
// directory, using the given argument vector.
// This function does not return on success.
void ExecuteProgram(const std::string& exe,
                    const std::vector<std::string>& args_vector);

class BlazeServerStartup {
 public:
  virtual ~BlazeServerStartup() {}
  virtual bool IsStillAlive() = 0;
};

// Starts a daemon process with its standard output and standard error
// redirected to the file "daemon_output". Sets server_startup to an object
// that can be used to query if the server is still alive. The PID of the
// daemon started is written into server_dir, both as a symlink (for legacy
// reasons) and as a file.
void ExecuteDaemon(const std::string& exe,
                   const std::vector<std::string>& args_vector,
                   const std::string& daemon_output,
                   const std::string& server_dir,
                   BlazeServerStartup** server_startup);

// Get the version string from the given java executable. The java executable
// is supposed to output a string in the form '.*version ".*".*'. This method
// will return the part in between the two quote or the empty string on failure
// to match the good string.
std::string GetJvmVersion(const std::string& java_exe);

// Convert a path from Bazel internal form to underlying OS form.
// On Unixes this is an identity operation.
// On Windows, Bazel internal form is cygwin path, and underlying OS form
// is Windows path.
std::string ConvertPath(const std::string& path);

// Convert a path list from Bazel internal form to underlying OS form.
// On Unixes this is an identity operation.
// On Windows, Bazel internal form is cygwin path list, and underlying OS form
// is Windows path list.
std::string ConvertPathList(const std::string& path_list);

// Converts `path` to a string that's safe to pass as path in a JVM flag.
// See https://github.com/bazelbuild/bazel/issues/2576
std::string PathAsJvmFlag(const std::string& path);

// A character used to separate paths in a list.
extern const char kListSeparator;

// Create a symlink to directory ``target`` at location ``link``.
// Returns true on success, false on failure. The target must be absolute.
// Implemented via junctions on Windows.
bool SymlinkDirectories(const std::string& target, const std::string& link);

// Compares two absolute paths. Necessary because the same path can have
// multiple different names under msys2: "C:\foo\bar" or "C:/foo/bar"
// (Windows-style) and "/c/foo/bar" (msys2 style). Returns if the paths are
// equal.
bool CompareAbsolutePaths(const std::string& a, const std::string& b);

struct BlazeLock {
#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
  /* HANDLE */ void* handle;
#else
  int lockfd;
#endif
};

// Acquires a lock on the output base. Exits if the lock cannot be acquired.
// Sets ``lock`` to a value that can subsequently be passed to ReleaseLock().
// Returns the number of milliseconds spent with waiting for the lock.
uint64_t AcquireLock(const std::string& output_base, bool batch_mode,
                     bool block, BlazeLock* blaze_lock);

// Releases the lock on the output base. In case of an error, continues as
// usual.
void ReleaseLock(BlazeLock* blaze_lock);

// Verifies whether the server process still exists. Returns true if it does.
bool VerifyServerProcess(int pid, const std::string& output_base);

// Kills a server process based on its PID.
// Returns true if the server process was found and killed.
// WARNING! This function can be called from a signal handler!
bool KillServerProcess(int pid, const std::string& output_base);

// Wait for approximately the specified number of milliseconds. The actual
// amount of time waited may be more or less because of interrupts or system
// clock resolution.
void TrySleep(unsigned int milliseconds);

// Mark path as being excluded from backups (if supported by operating system).
void ExcludePathFromBackup(const std::string& path);

// Returns the canonical form of the base dir given a root and a hashable
// string. The resulting dir is composed of the root + md5(hashable)
std::string GetHashedBaseDir(const std::string& root,
                             const std::string& hashable);

// Create a safe installation directory where we keep state, installations etc.
// This method ensures that the directory is created, is owned by the current
// user, and not accessible to anyone else.
void CreateSecureOutputRoot(const std::string& path);

std::string GetEnv(const std::string& name);

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

// Ensure we have open file descriptors for stdin/stdout/stderr.
void SetupStdStreams();

std::string GetUserName();

// Returns true iff the current terminal is running inside an Emacs.
bool IsEmacsTerminal();

// Returns true iff the current terminal can support color and cursor movement.
bool IsStandardTerminal();

// Returns the number of columns of the terminal to which stdout is
// connected, or 80 if there is no such terminal.
int GetTerminalColumns();

// Gets the system-wide explicit limit for the given resource.
//
// The resource is one of the RLIMIT_* constants defined in sys/resource.h.
// Returns 0 if the limit could not be fetched and returns -1 if the function
// is not implemented for this platform.
//
// It is OK to call this function with a parameter of -1 to check if the
// function is implemented for the platform.
int32_t GetExplicitSystemLimit(const int resource);

// Raises soft system resource limits to hard limits in an attempt to let
// large builds work. This is a best-effort operation and may or may not be
// implemented for a given platform. Returns true if all limits were properly
// raised; false otherwise.
bool UnlimitResources();

void DetectBashOrDie();

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_PLATFORM_H_
