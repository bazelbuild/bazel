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

// process-wrapper runs a subprocess with a given timeout (optional),
// redirecting stdout and stderr to given files. Upon exit, whether
// from normal termination or timeout, the subprocess (and any of its children)
// is killed.
//
// The exit status of this program is whatever the child process returned,
// unless process-wrapper receives a signal. ie, on SIGTERM this program will
// die with raise(SIGTERM) even if the child process handles SIGTERM with
// exit(0).

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// Not in headers on OSX.
extern char **environ;

static int global_pid;  // Returned from fork().
static int global_signal = -1;
static double global_kill_delay = 0.0;

#define DIE(args...) { \
  fprintf(stderr, args); \
  fprintf(stderr, " --- "); \
  perror(NULL); \
  fprintf(stderr, "\n"); \
  exit(EXIT_FAILURE); \
}

#define CHECK_CALL(x) if (x != 0) { perror(#x); exit(1); }

// Make sure the process and all subprocesses are killed.
static void KillEverything(int pgrp) {
  kill(-pgrp, SIGTERM);

  // Round up fractional seconds in this polling implementation.
  int kill_delay = (int)(global_kill_delay+0.999) ;
  // If the process is still alive, give it some time to die gracefully.
  while (kill(-pgrp, 0) == 0 && kill_delay-- > 0) {
    sleep(1);
  }

  kill(-pgrp, SIGKILL);
}

// Called when timeout or Signal occurs.
static void OnSignal(int sig) {
  global_signal = sig;
  if (sig == SIGALRM) {
    // SIGALRM represents a timeout, so we should give the process a bit of
    // time to die gracefully if it needs it.
    KillEverything(global_pid);
  } else {
    // Signals should kill the process quickly, as it's typically blocking
    // the return of the prompt after a user hits "Ctrl-C".
    kill(-global_pid, SIGKILL);
  }
}

// Set up a signal handler which kills all subprocesses when the
// given signal is triggered.
static void InstallSignalHandler(int sig) {
  struct sigaction sa = {};

  sa.sa_handler = OnSignal;
  sigemptyset(&sa.sa_mask);
  CHECK_CALL(sigaction(sig, &sa, NULL));
}

// Revert signal handler to default.
static void UnHandle(int sig) {
  struct sigaction sa = {};
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  CHECK_CALL(sigaction(sig, &sa, NULL));
}

// Enable the given timeout, or no-op if the timeout is non-positive.
static void EnableAlarm(double timeout) {
  if (timeout <= 0) return;

  struct itimerval timer = {};
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0;

  double int_val, fraction_val;
  fraction_val = modf(timeout, &int_val);
  timer.it_value.tv_sec = (long) int_val;
  timer.it_value.tv_usec = (long) (fraction_val * 1e6);
  CHECK_CALL(setitimer(ITIMER_REAL, &timer, NULL));
}

static void ClearSignalMask() {
  // Use an empty signal mask and default signal handlers in the
  // subprocess.
  sigset_t sset;
  sigemptyset(&sset);
  sigprocmask(SIG_SETMASK, &sset, NULL);
  for (int i = 1; i < NSIG; ++i) {
    if (i == SIGKILL || i == SIGSTOP) continue;

    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(i, &sa, NULL);
  }
}

static int WaitChild(pid_t pid, const char *name) {
  int err = 0;
  int status = 0;
  do {
    err = waitpid(pid, &status, 0);
  } while (err == -1 && errno == EINTR);

  if (err == -1) {
    DIE("wait on %s (pid %d) failed", name, pid);
  }
  return status;
}

// Usage: process-wrapper
//            <timeout_sec> <kill_delay_sec> <stdout file> <stderr file>
//            [cmdline]
int main(int argc, char *argv[]) {
  if (argc <= 5) {
    DIE("Not enough cmd line arguments to process-wrapper");
  }

  // Parse the cmdline args to get the timeout and redirect files.
  argv++;
  double timeout;
  if (sscanf(*argv++, "%lf", &timeout) != 1) {
    DIE("timeout_sec is not a real number.");
  }
  if (sscanf(*argv++, "%lf", &global_kill_delay) != 1) {
    DIE("kill_delay_sec is not a real number.");
  }
  char *stdout_path = *argv++;
  char *stderr_path = *argv++;

  if (strcmp(stdout_path, "-")) {
    // Redirect stdout and stderr.
    int fd_out = open(stdout_path, O_WRONLY|O_CREAT|O_TRUNC, 0666);
    if (fd_out == -1) {
      DIE("Could not open %s for stdout", stdout_path);
    }
    if (dup2(fd_out, STDOUT_FILENO) == -1) {
      DIE("dup2 failed for stdout");
    }
    CHECK_CALL(close(fd_out));
  }

  if (strcmp(stderr_path, "-")) {
    int fd_err = open(stderr_path, O_WRONLY|O_CREAT|O_TRUNC, 0666);
    if (fd_err == -1) {
      DIE("Could not open %s for stderr", stderr_path);
    }
    if (dup2(fd_err, STDERR_FILENO) == -1) {
      DIE("dup2 failed for stderr");
    }
    CHECK_CALL(close(fd_err));
  }

  global_pid = fork();
  if (global_pid < 0) {
    DIE("Fork failed");
  } else if (global_pid == 0) {
    // In child.
    if (setsid() == -1) {
      DIE("Could not setsid from child");
    }
    ClearSignalMask();
    // Force umask to include read and execute for everyone, to make
    // output permissions predictable.
    umask(022);

    execvp(argv[0], argv);  // Does not return.
    DIE("execvpe %s failed", argv[0]);
  } else {
    // In parent.
    InstallSignalHandler(SIGALRM);
    InstallSignalHandler(SIGTERM);
    InstallSignalHandler(SIGINT);
    EnableAlarm(timeout);

    int status = WaitChild(global_pid, argv[0]);

    // The child is done, but may have grandchildren.
    kill(-global_pid, SIGKILL);
    if (global_signal > 0) {
      // Don't trust the exit code if we got a timeout or signal.
      UnHandle(global_signal);
      raise(global_signal);
    } else if (WIFEXITED(status)) {
      exit(WEXITSTATUS(status));
    } else {
      int sig = WTERMSIG(status);
      UnHandle(sig);
      raise(sig);
    }
  }
}
