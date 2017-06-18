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

#include "src/main/tools/process-wrapper.h"

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
#include "src/main/tools/process-wrapper-legacy.h"

struct Options opt;

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

int main(int argc, char *argv[]) {
  std::vector<char *> args(argv, argv + argc);
  ParseCommandLine(args);

  SwitchToEuid();
  SwitchToEgid();

  Redirect(opt.stdout_path, STDOUT_FILENO);
  Redirect(opt.stderr_path, STDERR_FILENO);

  LegacyProcessWrapper::RunCommand();

  return 0;
}
