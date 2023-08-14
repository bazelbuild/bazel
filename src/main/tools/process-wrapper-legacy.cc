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

#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"
#include "src/main/tools/process-wrapper-options.h"
#include "src/main/tools/process-wrapper.h"

static bool child_subreaper_enabled = false;
#if defined(__linux__)
#if !defined(PR_SET_CHILD_SUBREAPER)
// https://github.com/torvalds/linux/blob/v5.7/tools/include/uapi/linux/prctl.h#L158
#define PR_SET_CHILD_SUBREAPER 36
#endif
#endif

pid_t LegacyProcessWrapper::child_pid = 0;
volatile sig_atomic_t LegacyProcessWrapper::last_signal = 0;

void LegacyProcessWrapper::RunCommand() {
  SpawnChild();
  WaitForChild();
}

void LegacyProcessWrapper::SpawnChild() {
#if defined(__linux__)
  if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0) {
    child_subreaper_enabled = true;
  } else {
    if (errno != EINVAL) {
      DIE("prctl");
    }
  }
#endif

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

// Sets up signal handlers to kill all subprocesses when the given signal is
// triggered. Whether subprocesses are abruptly terminated or not depends on
// the signal type and the user configuration.
void LegacyProcessWrapper::SetupSignalHandlers() {
  // SIGALRM represents a timeout so we should give the process a bit of time
  // to die gracefully if it needs it.
  InstallSignalHandler(SIGALRM, OnGracefulSignal);

  // Termination signals should kill the process quickly, as it's typically
  // blocking the return of the prompt after a user hits "Ctrl-C". But we allow
  // customizing the behavior of SIGTERM because it's used by the dynamic
  // scheduler to terminate process trees in a controlled manner.
  if (opt.graceful_sigterm) {
    InstallSignalHandler(SIGTERM, OnGracefulSignal);
  } else {
    InstallSignalHandler(SIGTERM, OnAbruptSignal);
  }
  InstallSignalHandler(SIGINT, OnAbruptSignal);
}

void LegacyProcessWrapper::WaitForChild() {
  SetupSignalHandlers();
  if (opt.timeout_secs > 0) {
    SetTimeout(opt.timeout_secs);
  }

  // On macOS, we have to ensure the whole process group is terminated before
  // collecting the status of the PID we are interested in. (Otherwise other
  // processes could race us and grab the PGID.)
#if defined(__APPLE__)
  if (WaitForProcessToTerminate(child_pid) == -1) {
    DIE("WaitForProcessToTerminate");
  }

  // The child is done for, but may have grandchildren that we still have to
  // kill.
  kill(-child_pid, SIGKILL);

  if (WaitForProcessGroupToTerminate(child_pid) == -1) {
    DIE("WaitForProcessGroupToTerminate");
  }
#endif

  int status;
  if (!opt.stats_path.empty()) {
    struct rusage child_rusage;
    status = WaitChildWithRusage(child_pid, &child_rusage,
                                 child_subreaper_enabled);
    WriteStatsToFile(&child_rusage, opt.stats_path);
  } else {
    status = WaitChild(child_pid, child_subreaper_enabled);
  }

#if !defined(__APPLE__) && !defined(__OpenBSD__)
  if (child_subreaper_enabled) {
    // If we enabled the child subreaper feature (on Linux), now that we have
    // collected the status of the PID we were interested in, terminate the
    // rest of the process group and wait until all the children are gone.
    //
    // If you are wondering why we don't use a PID namespace instead, it's
    // because those can have subtle effects on the processes we spawn (like
    // them assuming that the PIDs that they get are unique). The linux-sandbox
    // offers this functionality.
    if (TerminateAndWaitForAll(child_pid) == -1) {
      DIE("TerminateAndWaitForAll");
    }
  } else {
    // The child is done for, but may have grandchildren that we still have to
    // kill.
    kill(-child_pid, SIGKILL);
  }
#endif

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
void LegacyProcessWrapper::OnAbruptSignal(int sig) {
  last_signal = sig;
  KillEverything(child_pid, false, opt.kill_delay_secs);
}

// Called when timeout or signal occurs.
void LegacyProcessWrapper::OnGracefulSignal(int sig) {
  last_signal = sig;
  KillEverything(child_pid, true, opt.kill_delay_secs);
}
