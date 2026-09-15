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

#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/*
 * This process just kills itself will the signal number
 * specified on the command line.
 */

int main(int argc, char **argv) {
  /*
   * Parse command-line arguments.
   */
  const char *progname = argv[0] ? argv[0] : "killmyself";
  if (argc != 2) {
    fprintf(stderr, "%s: Usage: %s <signal-number>\n",
            progname, progname);
    exit(1);
  }
  int sig = atoi(argv[1]);

  /*
   * Restore the default signal action, in case the
   * parent process was ignoring the signal.
   *
   * This is needed because run_unittests ignores
   * SIGHUP signals and this gets inherited by child
   * processes.
   */
  signal(sig, SIG_DFL);

  /*
   * Send ourself the signal.
   */
  if (raise(sig) != 0) {
    fprintf(stderr, "%s: raise failed: %s", progname,
            strerror(errno));
    exit(1);
  }

  // We can get here if the signal was a non-fatal signal,
  // e.g. SIGCONT.
  exit(0);
}
