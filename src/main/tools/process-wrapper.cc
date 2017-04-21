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

#include "src/main/tools/process-tools.h"

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

using std::vector;

// Not in headers on OSX.
extern char **environ;

// The pid of the spawned child process.
static volatile sig_atomic_t global_child_pid;

// The signal that will be sent to the child when a timeout occurs.
static volatile sig_atomic_t global_next_timeout_signal = SIGTERM;

// Whether the child was killed due to a timeout.
static volatile sig_atomic_t global_timeout_occurred;

// Options parsing result.
struct Options {
  double timeout_secs;
  double kill_delay_secs;
  std::string stdout_path;
  std::string stderr_path;
  bool debug;
  vector<char *> args;
};

static struct Options opt;

// Print out a usage error and exit with EXIT_FAILURE.
static void Usage(char *program_name) {
  fprintf(stderr,
          "Usage: %s <timeout-secs> <kill-delay-secs> <stdout-redirect> "
          "<stderr-redirect> <command> [args] ...\n",
          program_name);
  exit(EXIT_FAILURE);
}

// Parse the command line flags and put the results in the global opt variable.
static void ParseCommandLine(vector<char *> args) {
  if (args.size() <= 5) {
    Usage(args.front());
  }

  int optind = 1;

  if (sscanf(args[optind++], "%lf", &opt.timeout_secs) != 1) {
    DIE("timeout_secs is not a real number.\n");
  }
  if (sscanf(args[optind++], "%lf", &opt.kill_delay_secs) != 1) {
    DIE("kill_delay_secs is not a real number.\n");
  }
  opt.stdout_path.assign(args[optind++]);
  opt.stderr_path.assign(args[optind++]);
  opt.args.assign(args.begin() + optind, args.end());

  // argv[] passed to execve() must be a null-terminated array.
  opt.args.push_back(nullptr);
}

static void OnTimeout(int signum) {
  global_timeout_occurred = true;
  kill(-global_child_pid, global_next_timeout_signal);
  if (global_next_timeout_signal == SIGTERM && opt.kill_delay_secs > 0) {
    global_next_timeout_signal = SIGKILL;
    SetTimeout(opt.kill_delay_secs);
  }
}

static void ForwardSignal(int signum) {
  if (global_child_pid > 0) {
    kill(-global_child_pid, signum);
  }
}

static void SetupSignalHandlers() {
  RestoreSignalHandlersAndMask();

  for (int signum = 1; signum < NSIG; signum++) {
    switch (signum) {
      // Some signals should indeed kill us and not be forwarded to the child,
      // thus we can use the default handler.
      case SIGABRT:
      case SIGBUS:
      case SIGFPE:
      case SIGILL:
      case SIGSEGV:
      case SIGSYS:
      case SIGTRAP:
        break;
      // It's fine to use the default handler for SIGCHLD, because we use wait()
      // in the main loop to wait for children to die anyway.
      case SIGCHLD:
        break;
      // One does not simply install a signal handler for these two signals
      case SIGKILL:
      case SIGSTOP:
        break;
      // Ignore SIGTTIN and SIGTTOU, as we hand off the terminal to the child in
      // SpawnChild().
      case SIGTTIN:
      case SIGTTOU:
        IgnoreSignal(signum);
        break;
      // We need a special signal handler for this if we use a timeout.
      case SIGALRM:
        if (opt.timeout_secs > 0) {
          InstallSignalHandler(signum, OnTimeout);
        } else {
          InstallSignalHandler(signum, ForwardSignal);
        }
        break;
      // All other signals should be forwarded to the child.
      default:
        InstallSignalHandler(signum, ForwardSignal);
        break;
    }
  }
}

int main(int argc, char *argv[]) {
  KillMeWhenMyParentDies(SIGTERM);
  DropPrivileges();

  vector<char *> args(argv, argv + argc);
  ParseCommandLine(args);

  Redirect(opt.stdout_path, STDOUT_FILENO);
  Redirect(opt.stderr_path, STDERR_FILENO);

  SetupSignalHandlers();
  BecomeSubreaper();
  global_child_pid = SpawnCommand(opt.args);

  if (opt.timeout_secs > 0) {
    SetTimeout(opt.timeout_secs);
  }

  int exitcode = WaitForChild(global_child_pid);
  if (global_timeout_occurred) {
    return 128 + SIGALRM;
  }

  return exitcode;
}
