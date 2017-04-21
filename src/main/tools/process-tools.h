// Copyright 2015 The Bazel Authors. All rights reserved.
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

#ifndef PROCESS_TOOLS_H__
#define PROCESS_TOOLS_H__

#include <string>
#include <vector>

#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)

#define DIE(...)                                                \
  {                                                             \
    fprintf(stderr, __FILE__ ":" S__LINE__ ": \"" __VA_ARGS__); \
    fprintf(stderr, "\": ");                                    \
    perror(nullptr);                                            \
    exit(EXIT_FAILURE);                                         \
  }

#define PRINT_DEBUG(...)                                        \
  do {                                                          \
    if (opt.debug) {                                            \
      fprintf(stderr, __FILE__ ":" S__LINE__ ": " __VA_ARGS__); \
      fprintf(stderr, "\n");                                    \
    }                                                           \
  } while (0)

// Set the effective and saved uid / gid to the real uid / gid.
void DropPrivileges();

// Redirect the open file descriptor fd to the file target_path. Do nothing if
// target_path is '-'.
void Redirect(const std::string &target_path, int fd);

// Write formatted contents into the file filename.
void WriteFile(const std::string &filename, const char *fmt, ...);

// Receive SIGALRM after the given timeout. timeout_secs must be positive.
void SetTimeout(double timeout_secs);

// Installs a signal handler for signum and sets all signals to block during
// that signal.
void InstallSignalHandler(int signum, void (*handler)(int));

// Sets the signal handler of signum to SIG_IGN.
void IgnoreSignal(int signum);

// Reset the signal mask and restore the default handler for all signals.
void RestoreSignalHandlersAndMask();

// Ask the kernel to kill us with signum if our parent dies.
void KillMeWhenMyParentDies(int signum);

// This is the magic that makes waiting for all children (even grandchildren)
// work. By becoming a subreaper, all grandchildren that are not waited for by
// our direct child will be reparented to us, which allows us to wait for them.
void BecomeSubreaper();

// Forks and execvp's the process specified in args in its own process group.
// Returns the pid of the spawned process.
int SpawnCommand(const std::vector<char *> &args);

// Waits for child_pid to exit, then kills all remaining (grand)children, waits
// for them to exit, then returns the exitcode of child_pid.
int WaitForChild(int child_pid);

#endif  // PROCESS_TOOLS_H__
