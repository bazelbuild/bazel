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

#include "src/main/tools/process-tools.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/main/tools/logging.h"

int SwitchToEuid() {
  int uid = getuid();
  int euid = geteuid();
  if (uid != euid) {
    if (setreuid(euid, euid) < 0) {
      DIE("setreuid");
    }
  }
  return euid;
}

int SwitchToEgid() {
  int gid = getgid();
  int egid = getegid();
  if (gid != egid) {
    if (setregid(egid, egid) < 0) {
      DIE("setregid");
    }
  }
  return egid;
}

void Redirect(const std::string &target_path, int fd) {
  if (!target_path.empty() && target_path != "-") {
    const int flags = O_WRONLY | O_CREAT | O_TRUNC | O_APPEND;
    int fd_out = open(target_path.c_str(), flags, 0666);
    if (fd_out < 0) {
      DIE("open(%s)", target_path.c_str());
    }
    // If we were launched with less than 3 fds (stdin, stdout, stderr) open,
    // but redirection is still requested via a command-line flag, something is
    // wacky and the following code would not do what we intend to do, so let's
    // bail.
    if (fd_out < 3) {
      DIE("open(%s) returned a handle that is reserved for stdin / stdout / "
          "stderr",
          target_path.c_str());
    }
    if (dup2(fd_out, fd) < 0) {
      DIE("dup2");
    }
    if (close(fd_out) < 0) {
      DIE("close");
    }
  }
}

void KillEverything(pid_t pgrp, bool gracefully, double graceful_kill_delay) {
  if (gracefully) {
    kill(-pgrp, SIGTERM);

    // Round up fractional seconds in this polling implementation.
    int kill_delay = static_cast<int>(ceil(graceful_kill_delay));

    // If the process is still alive, give it some time to die gracefully.
    while (kill_delay-- > 0 && kill(-pgrp, 0) == 0) {
      sleep(1);
    }
  }

  kill(-pgrp, SIGKILL);
}

void HandleSignal(int sig, void (*handler)(int)) {
  struct sigaction sa = {};
  sa.sa_handler = handler;
  if (sigemptyset(&sa.sa_mask) < 0) {
    DIE("sigemptyset");
  }
  if (sigaction(sig, &sa, nullptr) < 0) {
    DIE("sigaction");
  }
}

void UnHandle(int sig) {
  switch (sig) {
    case SIGSTOP:
    case SIGKILL:
      // These signals can't be handled, so they'll always have a valid default
      // handler. In fact, even trying to install SIG_DFL again will result in
      // EINVAL, so we'll just not do anything for these.
      return;
    default:
      HandleSignal(sig, SIG_DFL);
  }
}

void ClearSignalMask() {
  // Use an empty signal mask for the process.
  sigset_t empty_sset;
  if (sigemptyset(&empty_sset) < 0) {
    DIE("sigemptyset");
  }
  if (sigprocmask(SIG_SETMASK, &empty_sset, nullptr) < 0) {
    DIE("sigprocmask");
  }

  // Set the default signal handler for all signals.
  for (int i = 1; i < NSIG; ++i) {
    if (i == SIGKILL || i == SIGSTOP) {
      continue;
    }
    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    if (sigemptyset(&sa.sa_mask) < 0) {
      DIE("sigemptyset");
    }
    // Ignore possible errors, because we might not be allowed to set the
    // handler for certain signals, but we still want to try.
    sigaction(i, &sa, nullptr);
  }
}

void SetTimeout(double timeout_secs) {
  double int_val, fraction_val;
  fraction_val = modf(timeout_secs, &int_val);

  struct itimerval timer;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0;
  timer.it_value.tv_sec = static_cast<time_t>(int_val),
  timer.it_value.tv_usec = static_cast<suseconds_t>(fraction_val * 1e6);

  if (setitimer(ITIMER_REAL, &timer, nullptr) < 0) {
    DIE("setitimer");
  }
}

int WaitChild(pid_t pid) {
  int err, status;

  do {
    err = waitpid(pid, &status, 0);
  } while (err == -1 && errno == EINTR);

  if (err == -1) {
    DIE("wait");
  }

  return status;
}
