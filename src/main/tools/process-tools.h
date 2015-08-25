// Copyright 2015 Google Inc. All rights reserved.
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

#include <sys/types.h>
#include <stdbool.h>

// see
// http://stackoverflow.com/questions/5641427/how-to-make-preprocessor-generate-a-string-for-line-keyword
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)

#define DIE(args...)                                   \
  {                                                    \
    fprintf(stderr, __FILE__ ":" S__LINE__ ": " args); \
    exit(EXIT_FAILURE);                                \
  }

#define CHECK_CALL(x, ...)                                    \
  if ((x) == -1) {                                            \
    fprintf(stderr, __FILE__ ":" S__LINE__ ": " __VA_ARGS__); \
    perror(#x);                                               \
    exit(EXIT_FAILURE);                                       \
  }

#define CHECK_NOT_NULL(x) \
  if (x == NULL) {        \
    perror(#x);           \
    exit(EXIT_FAILURE);   \
  }

// Switch completely to the effective uid.
// Some programs (notably, bash) ignore the euid and just use the uid. This
// limits the ability for us to use process-wrapper as a setuid binary for
// security/user-isolation.
int SwitchToEuid();

// Switch completely to the effective gid.
int SwitchToEgid();

// Redirect stdout to the file stdout_path (but not if stdout_path is "-").
void RedirectStdout(const char *stdout_path);

// Redirect stderr to the file stdout_path (but not if stderr_path is "-").
void RedirectStderr(const char *stderr_path);

// Make sure the process group "pgrp" and all its subprocesses are killed.
// If "gracefully" is true, sends SIGKILL first and after a timeout of
// "graceful_kill_delay" seconds, sends SIGTERM.
// If not, send SIGTERM immediately.
void KillEverything(int pgrp, bool gracefully, double graceful_kill_delay);

// Set up a signal handler for a signal.
void HandleSignal(int sig, void (*handler)(int));

// Revert signal handler for a signal to the default.
void UnHandle(int sig);

// Use an empty signal mask for the process and set all signal handlers to their
// default.
void ClearSignalMask();

// Receive SIGALRM after the given timeout. No-op if the timeout is
// non-positive.
void SetTimeout(double timeout_secs);

// Wait for "pid" to exit and return its exit code.
// "name" is used for the error message only.
int WaitChild(pid_t pid, const char *name);

#endif  // PROCESS_TOOLS_H__
