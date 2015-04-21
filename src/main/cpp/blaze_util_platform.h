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

#ifndef DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_PLATFORM_H_
#define DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_PLATFORM_H_

#include <string>
#include "src/main/cpp/util/numbers.h"

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
// as a uint64 nanoseconds since epoch.
uint64 MonotonicClock();

// Wrapper around clock_gettime(CLOCK_PROCESS_CPUTIME_ID) that returns the
// nanoseconds consumed by the current process since it started.
uint64 ProcessClock();

// Set cpu and IO scheduling properties. Note that this can take ~50ms
// on Linux, so it should only be called when necessary.
void SetScheduling(bool batch_cpu_scheduling, int io_nice_level);

// Returns the cwd for a process.
std::string GetProcessCWD(int pid);

bool IsSharedLibrary(const std::string& filename);

// Return the default path to the JDK used to run Blaze itself
// (must be an absolute directory).
std::string GetDefaultHostJavabase();

}  // namespace blaze

#endif  // DEVTOOLS_BLAZE_MAIN_BLAZE_UTIL_PLATFORM_H_
