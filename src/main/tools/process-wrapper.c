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
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "process-tools.h"

// Not in headers on OSX.
extern char **environ;

static double global_kill_delay;
static int global_child_pid;
static volatile sig_atomic_t global_signal;

// Options parsing result.
struct Options {
  double timeout_secs;
  double kill_delay_secs;
  const char *stdout_path;
  const char *stderr_path;
  char *const *args;
};

// Print out a usage error. argc and argv are the argument counter and vector,
// fmt is a format,
// string for the error message to print.
static void Usage(char *const *argv) {
  fprintf(stderr,
          "Usage: %s <timeout-secs> <kill-delay-secs> <stdout-redirect> "
          "<stderr-redirect> <command> [args] ...\n",
          argv[0]);
  exit(EXIT_FAILURE);
}

// Parse the command line flags and return the result in an Options structure
// passed as argument.
static void ParseCommandLine(int argc, char *const *argv, struct Options *opt) {
  if (argc <= 5) {
    Usage(argv);
  }

  argv++;
  if (sscanf(*argv++, "%lf", &opt->timeout_secs) != 1) {
    DIE("timeout_secs is not a real number.\n");
  }
  if (sscanf(*argv++, "%lf", &opt->kill_delay_secs) != 1) {
    DIE("kill_delay_secs is not a real number.\n");
  }
  opt->stdout_path = *argv++;
  opt->stderr_path = *argv++;
  opt->args = argv;
}

// Called when timeout or signal occurs.
void OnSignal(int sig) {
  global_signal = sig;

  // Nothing to do if we received a signal before spawning the child.
  if (global_child_pid == -1) {
    return;
  }

  if (sig == SIGALRM) {
    // SIGALRM represents a timeout, so we should give the process a bit of
    // time to die gracefully if it needs it.
    KillEverything(global_child_pid, true, global_kill_delay);
  } else {
    // Signals should kill the process quickly, as it's typically blocking
    // the return of the prompt after a user hits "Ctrl-C".
    KillEverything(global_child_pid, false, global_kill_delay);
  }
}

// Run the command specified by the argv array and kill it after timeout
// seconds.
static void SpawnCommand(char *const *argv, double timeout_secs) {
  CHECK_CALL(global_child_pid = fork());
  if (global_child_pid == 0) {
    // In child.
    CHECK_CALL(setsid());
    ClearSignalMask();

    // Force umask to include read and execute for everyone, to make
    // output permissions predictable.
    umask(022);

    // Does not return unless something went wrong.
    CHECK_CALL(execvp(argv[0], argv));
  } else {
    // In parent.

    // Set up a signal handler which kills all subprocesses when the given
    // signal is triggered.
    HandleSignal(SIGALRM, OnSignal);
    HandleSignal(SIGTERM, OnSignal);
    HandleSignal(SIGINT, OnSignal);
    SetTimeout(timeout_secs);

    int status = WaitChild(global_child_pid, argv[0]);

    // The child is done for, but may have grandchildren that we still have to
    // kill.
    kill(-global_child_pid, SIGKILL);

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

int main(int argc, char *argv[]) {
  struct Options opt;
  memset(&opt, 0, sizeof(opt));

  ParseCommandLine(argc, argv, &opt);
  global_kill_delay = opt.kill_delay_secs;

  SwitchToEuid();
  SwitchToEgid();

  RedirectStdout(opt.stdout_path);
  RedirectStderr(opt.stderr_path);

  SpawnCommand(opt.args, opt.timeout_secs);

  return 0;
}
