// Copyright 2019 The Bazel Authors. All rights reserved.
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

#include <errno.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/main/tools/process-tools.h"

int TerminateAndWaitForAll(pid_t pid) {
  kill(-pid, SIGKILL);

  int res;
  while ((res = waitpid(-1, nullptr, WNOHANG)) > 0) {
    // Got one child; try again.
  }
  if (res == -1) {
    // The fast path got all children, so there is nothing else to do.
    return 0;
  }

  // Cope with children that may have escaped the process group or that
  // did not exit quickly enough.
  FILE *f = fopen("/proc/thread-self/children", "r");
  if (f == nullptr) {
    // Oh oh. This feature may be disabled, in which case there is
    // nothing we can do. Stop early and let any stale children be
    // reparented to init.
    return 0;
  }
  setbuf(f, nullptr);
  int child_pid;
  while ((waitpid(-1, nullptr, WNOHANG) != -1 || errno != ECHILD) &&
         (rewind(f), 1 == fscanf(f, "%d", &child_pid))) {
    kill(child_pid, SIGKILL);
    usleep(100);
  }
  fclose(f);

  return 0;
}
