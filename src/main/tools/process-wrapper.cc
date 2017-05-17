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

#include <err.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"

static double global_kill_delay;
static pid_t global_child_pid;
static volatile sig_atomic_t global_signal;

// Options parsing result.
struct Options {
  double timeout_secs;
  double kill_delay_secs;
  std::string stdout_path;
  std::string stderr_path;
  std::vector<char *> args;
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

// Parse the command line flags and return the result in an Options structure
// passed as argument.
static void ParseCommandLine(std::vector<char *> args) {
  if (args.size() < 5) {
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
static void SpawnCommand(const std::vector<char *> &args, double timeout_secs) {
  global_child_pid = fork();
  if (global_child_pid < 0) {
    DIE("fork");
  } else if (global_child_pid == 0) {
    // In child.
    if (setsid() < 0) {
      DIE("setsid");
    }
    ClearSignalMask();

    // Force umask to include read and execute for everyone, to make
    // output permissions predictable.
    umask(022);

    // Does not return unless something went wrong.
    if (execvp(args[0], args.data()) < 0) {
      DIE("execvp(%s, ...)", args[0]);
    }
  } else {
    // In parent.

    // Set up a signal handler which kills all subprocesses when the given
    // signal is triggered.
    InstallSignalHandler(SIGALRM, OnSignal);
    InstallSignalHandler(SIGTERM, OnSignal);
    InstallSignalHandler(SIGINT, OnSignal);
    if (timeout_secs > 0) {
      SetTimeout(timeout_secs);
    }

    int status = WaitChild(global_child_pid);

    // The child is done for, but may have grandchildren that we still have to
    // kill.
    kill(-global_child_pid, SIGKILL);

    if (global_signal > 0) {
      // Don't trust the exit code if we got a timeout or signal.
      InstallDefaultSignalHandler(global_signal);
      raise(global_signal);
    } else if (WIFEXITED(status)) {
      exit(WEXITSTATUS(status));
    } else {
      int sig = WTERMSIG(status);
      InstallDefaultSignalHandler(sig);
      raise(sig);
    }
  }
}

int main(int argc, char *argv[]) {
  std::vector<char *> args(argv, argv + argc);
  ParseCommandLine(args);
  global_kill_delay = opt.kill_delay_secs;

  SwitchToEuid();
  SwitchToEgid();

  Redirect(opt.stdout_path, STDOUT_FILENO);
  Redirect(opt.stderr_path, STDERR_FILENO);

  SpawnCommand(opt.args, opt.timeout_secs);

  return 0;
}
