// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include "src/main/tools/process-wrapper-legacy.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"
#include "src/main/tools/process-wrapper-options.h"
#include "src/main/tools/process-wrapper.h"

pid_t LegacyProcessWrapper::child_pid = 0;
volatile sig_atomic_t LegacyProcessWrapper::last_signal = 0;

void LegacyProcessWrapper::RunCommand() {
  SpawnChild();
  WaitForChild();
}

void LegacyProcessWrapper::SpawnChild() {
  child_pid = fork();
  if (child_pid < 0) {
    DIE("fork");
  } else if (child_pid == 0) {
    // In child.
    if (setsid() < 0) {
      DIE("setsid");
    }
    ClearSignalMask();

    // Force umask to include read and execute for everyone, to make output
    // permissions predictable.
    umask(022);

    // Does not return unless something went wrong.
    if (execvp(opt.args[0], opt.args.data()) < 0) {
      DIE("execvp(%s, ...)", opt.args[0]);
    }
  }
}

void LegacyProcessWrapper::WaitForChild() {
  // Set up a signal handler which kills all subprocesses when the given signal
  // is triggered.
  InstallSignalHandler(SIGALRM, OnSignal);
  InstallSignalHandler(SIGTERM, OnSignal);
  InstallSignalHandler(SIGINT, OnSignal);
  if (opt.timeout_secs > 0) {
    SetTimeout(opt.timeout_secs);
  }

  int status;
  if (!opt.stats_path.empty()) {
    struct rusage child_rusage;
    status = WaitChildWithRusage(child_pid, &child_rusage);
    WriteStatsToFile(&child_rusage, opt.stats_path);
  } else {
    status = WaitChild(child_pid);
  }

  // The child is done for, but may have grandchildren that we still have to
  // kill.
  kill(-child_pid, SIGKILL);

  if (last_signal > 0) {
    // Don't trust the exit code if we got a timeout or signal.
    InstallDefaultSignalHandler(last_signal);
    raise(last_signal);
  } else if (WIFEXITED(status)) {
    exit(WEXITSTATUS(status));
  } else {
    int sig = WTERMSIG(status);
    InstallDefaultSignalHandler(sig);
    raise(sig);
  }
}

// Called when timeout or signal occurs.
void LegacyProcessWrapper::OnSignal(int sig) {
  last_signal = sig;

  if (sig == SIGALRM) {
    // SIGALRM represents a timeout, so we should give the process a bit of time
    // to die gracefully if it needs it.
    KillEverything(child_pid, true, opt.kill_delay_secs);
  } else {
    // Signals should kill the process quickly, as it's typically blocking the
    // return of the prompt after a user hits "Ctrl-C".
    KillEverything(child_pid, false, opt.kill_delay_secs);
  }
}
