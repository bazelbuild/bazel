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
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/event.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"

namespace {

int WaitForProcessToTerminate(uintptr_t ident) {
  int kq;
  if ((kq = kqueue()) == -1) {
    return -1;
  }

  // According to the kqueue(2) documentation, registering for an event
  // reports any pending such events, so this is not racy even if the
  // process happened to exit before we got to installing the kevent.
  struct kevent kc;
  EV_SET(&kc, ident, EVFILT_PROC, EV_ADD | EV_ENABLE, NOTE_EXIT, 0, 0);

  int nev;
  struct kevent ke;
retry:
  if ((nev = kevent(kq, &kc, 1, &ke, 1, nullptr)) == -1) {
    if (errno == EINTR) {
      goto retry;
    }
    return -1;
  }
  if (nev != 1) {
    DIE("Expected only one event from the kevent call; got %d", nev);
  }
  if (ke.ident != ident) {
    DIE("Expected PID in the kevent to be %" PRIdMAX " but got %" PRIdMAX,
        (intmax_t)ident, (intmax_t)ke.ident);
  }
  if (!(ke.fflags & NOTE_EXIT)) {
    DIE("Expected the kevent to be for an exit condition");
  }

  return close(kq);
}

}  // namespace

int WaitForProcessToTerminate(pid_t pid) {
  if (pid < 0) {
    DIE("PID must be >= 0, got %" PRIdMAX, static_cast<intmax_t>(pid));
  }

  return WaitForProcessToTerminate((uintptr_t)pid);
}

int WaitForProcessGroupToTerminate(pid_t pgid) {
  int name[] = {CTL_KERN, KERN_PROC, KERN_PROC_PGRP, pgid};

  for (;;) {
    // Query the list of processes in the group by using sysctl(3).
    // This is "hard" because we don't know how big that list is, so we
    // have to first query the size of the output data and then account for
    // the fact that the size might change by the time we actually issue
    // the query.
    struct kinfo_proc *procs = nullptr;
    size_t nprocs = 0;
    do {
      size_t len;
      if (sysctl(name, 4, 0, &len, nullptr, 0) == -1) {
        return -1;
      }
      procs = (struct kinfo_proc *)malloc(len);
      if (sysctl(name, 4, procs, &len, nullptr, 0) == -1) {
        if (errno != ENOMEM) {
          DIE("Unexpected error code %d", errno);
        }
        free(procs);
        procs = nullptr;
      } else {
        nprocs = len / sizeof(struct kinfo_proc);
      }
    } while (procs == nullptr);
    if (nprocs < 1) {
      DIE("Must have found the group leader at least");
    }

    if (nprocs == 1) {
      // Found only one process, which must be the leader because we have
      // purposely expect it as a zombie with WaitForProcess.
#if defined(__OpenBSD__)
      if (procs->p_pid != pgid) {
#else
      if (procs->kp_proc.p_pid != pgid) {
#endif
        DIE("Process group leader must be the only process left");
      }
      free(procs);
      return 0;
    }
    free(procs);

    // More than one process left in the process group.  Kill the group
    // again just in case any extra processes appeared just now, which
    // would not allow us to complete quickly.
    kill(-pgid, SIGKILL);

    // And pause a little bit before retrying to avoid burning CPU.
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 1000000;
    if (nanosleep(&ts, nullptr) == -1) {
      return -1;
    }
  }
}
