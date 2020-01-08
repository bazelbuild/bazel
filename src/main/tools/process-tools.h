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

#ifndef SRC_MAIN_TOOLS_PROCESS_TOOLS_H_
#define SRC_MAIN_TOOLS_PROCESS_TOOLS_H_

#include <stdbool.h>
#include <sys/types.h>
#include <string>

// Switch completely to the effective uid.
// Some programs (notably, bash) ignore the euid and just use the uid. This
// limits the ability for us to use process-wrapper as a setuid binary for
// security/user-isolation.
int SwitchToEuid();

// Switch completely to the effective gid.
int SwitchToEgid();

// Redirect fd to the file target_path (but not if target_path is empty or "-").
void Redirect(const std::string &target_path, int fd);

// Make sure the process group "pgrp" and all its subprocesses are killed.
// If "gracefully" is true, sends SIGTERM first and after a timeout of
// "graceful_kill_delay" seconds, sends SIGKILL.
// If not, send SIGKILL immediately.
void KillEverything(pid_t pgrp, bool gracefully, double graceful_kill_delay);

// Set up a signal handler for a signal.
void InstallSignalHandler(int signum, void (*handler)(int));

// Set the signal handler for `signum` to SIG_IGN (ignore).
void IgnoreSignal(int signum);

// Set the signal handler for `signum` to SIG_DFL (default).
void InstallDefaultSignalHandler(int sig);

// Use an empty signal mask for the process and set all signal handlers to their
// default.
void ClearSignalMask();

// Receive SIGALRM after the given timeout. No-op if the timeout is
// non-positive.
void SetTimeout(double timeout_secs);

// Wait for "pid" to exit and return its exit code.
int WaitChild(pid_t pid);

// Wait for "pid" to exit and return its exit code.
// Resource usage is returned in "rusage" regardless of the exit status of the
// child process.
int WaitChildWithRusage(pid_t pid, struct rusage *rusage);

// Write execution statistics to a file.
void WriteStatsToFile(struct rusage *rusage, const std::string &stats_path);

#endif  // PROCESS_TOOLS_H__
