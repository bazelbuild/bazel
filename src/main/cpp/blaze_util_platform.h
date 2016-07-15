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

namespace blaze {

// Get the absolute path to the binary being executed.
std::string GetSelfPath();

// Returns the directory Bazel can use to store output.
std::string GetOutputRoot();

// Returns the process id of the peer connected to this socket.
pid_t GetPeerProcessId(int socket);

// Warn about dubious filesystem types, such as NFS, case-insensitive (?).
void WarnFilesystemType(const std::string& output_base);

// Wrapper around clock_gettime(CLOCK_MONOTONIC) that returns the time
// as a uint64_t nanoseconds since epoch.
uint64_t MonotonicClock();

// Wrapper around clock_gettime(CLOCK_PROCESS_CPUTIME_ID) that returns the
// nanoseconds consumed by the current process since it started.
uint64_t ProcessClock();

// Set cpu and IO scheduling properties. Note that this can take ~50ms
// on Linux, so it should only be called when necessary.
void SetScheduling(bool batch_cpu_scheduling, int io_nice_level);

// Returns the cwd for a process.
std::string GetProcessCWD(int pid);

bool IsSharedLibrary(const std::string& filename);

// Return the default path to the JDK used to run Blaze itself
// (must be an absolute directory).
std::string GetDefaultHostJavabase();

// Replace the current process with the given program in the current working
// directory, using the given argument vector.
// This function does not return on success.
void ExecuteProgram(const string& exe, const std::vector<string>& args_vector);

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
void ExecuteDaemon(const string& exe, const std::vector<string>& args_vector,
                   const string& daemon_output, const string& server_dir,
                   BlazeServerStartup** server_startup);

// Executes a subprocess and returns its standard output and standard error.
// If this fails, exits with the appropriate error code.
string RunProgram(const string& exe, const std::vector<string>& args_vector);

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

// Return a string used to separate paths in a list.
std::string ListSeparator();

// Create a symlink to directory ``target`` at location ``link``.
// Returns true on success, false on failure. The target must be absolute.
// Implemented via junctions on Windows.
bool SymlinkDirectories(const string &target, const string &link);

// Reads which directory a symlink points to. Puts the target of the symlink
// in ``result`` and returns if the operation was successful. Will not work on
// symlinks that don't point to directories on Windows.
bool ReadDirectorySymlink(const string &symlink, string *result);

// Compares two absolute paths. Necessary because the same path can have
// multiple different names under msys2: "C:\foo\bar" or "C:/foo/bar"
// (Windows-style) and "/c/foo/bar" (msys2 style). Returns if the paths are
// equal.
bool CompareAbsolutePaths(const string& a, const string& b);

struct BlazeLock {
  int lockfd;
};

// Acquires a lock on the output base. Exits if the lock cannot be acquired.
// Sets ``lock`` to a value that can subsequently be passed to ReleaseLock().
// Returns the number of milliseconds spent with waiting for the lock.
uint64_t AcquireLock(const string& output_base, bool batch_mode,
                     bool block, BlazeLock* blaze_lock);

// Releases the lock on the output base. In case of an error, continues as
// usual.
void ReleaseLock(BlazeLock* blaze_lock);

// Kills a server process based on its output base and PID. Returns true if the
// server process was found and killed.
// This function can be called from a signal handler!
bool KillServerProcess(
    int pid, const string& output_base, const string& install_base);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_UTIL_PLATFORM_H_
