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

#define _GNU_SOURCE

#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#include "process-tools.h"

int SwitchToEuid() {
  int uid = getuid();
  int euid = geteuid();
  if (uid != euid) {
    CHECK_CALL(setreuid(euid, euid));
  }
  return euid;
}

int SwitchToEgid() {
  int gid = getgid();
  int egid = getegid();
  if (gid != egid) {
    CHECK_CALL(setregid(egid, egid));
  }
  return egid;
}

void Redirect(const char *target_path, int fd, const char *name) {
  if (target_path != NULL && strcmp(target_path, "-") != 0) {
    int fd_out;
    const int flags = O_WRONLY | O_CREAT | O_TRUNC | O_APPEND;
    CHECK_CALL(fd_out = open(target_path, flags, 0666));
    CHECK_CALL(dup2(fd_out, fd));
    CHECK_CALL(close(fd_out));
  }
}

void RedirectStdout(const char *stdout_path) {
  Redirect(stdout_path, STDOUT_FILENO, "stdout");
}

void RedirectStderr(const char *stderr_path) {
  Redirect(stderr_path, STDERR_FILENO, "stderr");
}

void KillEverything(int pgrp, bool gracefully, double graceful_kill_delay) {
  if (gracefully) {
    kill(-pgrp, SIGTERM);

    // Round up fractional seconds in this polling implementation.
    int kill_delay = (int)(ceil(graceful_kill_delay));

    // If the process is still alive, give it some time to die gracefully.
    while (kill_delay-- > 0 && kill(-pgrp, 0) == 0) {
      sleep(1);
    }
  }

  kill(-pgrp, SIGKILL);
}

void HandleSignal(int sig, void (*handler)(int)) {
  struct sigaction sa = {.sa_handler = handler};
  CHECK_CALL(sigemptyset(&sa.sa_mask));
  CHECK_CALL(sigaction(sig, &sa, NULL));
}

void UnHandle(int sig) { HandleSignal(sig, SIG_DFL); }

void ClearSignalMask() {
  // Use an empty signal mask for the process.
  sigset_t empty_sset;
  CHECK_CALL(sigemptyset(&empty_sset));
  CHECK_CALL(sigprocmask(SIG_SETMASK, &empty_sset, NULL));

  // Set the default signal handler for all signals.
  for (int i = 1; i < NSIG; ++i) {
    if (i == SIGKILL || i == SIGSTOP) {
      continue;
    }
    struct sigaction sa = {.sa_handler = SIG_DFL};
    CHECK_CALL(sigemptyset(&sa.sa_mask));
    // Ignore possible errors, because we might not be allowed to set the
    // handler for certain signals, but we still want to try.
    sigaction(i, &sa, NULL);
  }
}

void SetTimeout(double timeout_secs) {
  if (timeout_secs <= 0) {
    return;
  }

  double int_val, fraction_val;
  fraction_val = modf(timeout_secs, &int_val);

  struct itimerval timer;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0;
  timer.it_value.tv_sec = (long)int_val,
  timer.it_value.tv_usec = (long)(fraction_val * 1e6);

  CHECK_CALL(setitimer(ITIMER_REAL, &timer, NULL));
}

int WaitChild(pid_t pid, const char *name) {
  int err, status;

  do {
    err = waitpid(pid, &status, 0);
  } while (err == -1 && errno == EINTR);

  if (err == -1) {
    DIE("wait on %s (pid %d) failed\n", name, pid);
  }

  return status;
}
