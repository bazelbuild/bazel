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

#ifndef SRC_MAIN_TOOLS_PROCESS_WRAPPER_LEGACY_H_
#define SRC_MAIN_TOOLS_PROCESS_WRAPPER_LEGACY_H_

#include <signal.h>
#include <vector>

// The process-wrapper implementation that was used until and including Bazel
// 0.4.5. Waits for the wrapped process to exit and then kills its process
// group. Works on all POSIX operating systems (tested on Linux, macOS,
// FreeBSD, and OpenBSD).
//
// Caveats:
// - Killing just the process group of the spawned child means that daemons or
//   other processes spawned by the child may not be killed if they change their
//   process group.
// - Does not wait for grandchildren to exit, thus processes spawned by the
//   child that could not be killed will linger around in the background.
// - Has a PID reuse race condition, because the kill() to the process group is
//   sent after waitpid() was called on the main child.
class LegacyProcessWrapper {
 public:
  // Run the command specified in the `opt.args` array and kill it after
  // `opt.timeout_secs` seconds.
  static void RunCommand();

 private:
  static void SpawnChild();
  static void WaitForChild();
  static void OnSignal(int sig);

  static pid_t child_pid;
  static volatile sig_atomic_t last_signal;
};

#endif
