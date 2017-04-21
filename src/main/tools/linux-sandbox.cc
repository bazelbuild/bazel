// Copyright 2016 The Bazel Authors. All rights reserved.
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

/**
 * linux-sandbox runs commands in a restricted environment where they are
 * subject to a few rules:
 *
 *  - The entire filesystem is made read-only.
 *  - The working directory (-W) will be made read-write, though.
 *  - Individual files or directories can be made writable (but not deletable)
 *    (-w).
 *  - If the process takes longer than the timeout (-T), it will be killed with
 *    SIGTERM. If it does not exit within the grace period (-t), it all of its
 *    children will be killed with SIGKILL.
 *  - tmpfs can be mounted on top of existing directories (-e).
 *  - If option -R is passed, the process will run as user 'root'.
 *  - If option -U is passed, the process will run as user 'nobody'.
 *  - Otherwise, the process runs using the current uid / gid.
 *  - If linux-sandbox itself gets killed, the process and all of its children
 *    will be killed.
 *  - If linux-sandbox's parent dies, it will kill itself, the process and all
 *    the children.
 *  - Network access is allowed, but can be disabled via -N.
 *  - The hostname and domainname will be set to "sandbox".
 *  - The process runs in its own PID namespace, so other processes on the
 *    system are invisible.
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "src/main/tools/linux-sandbox-options.h"
#include "src/main/tools/linux-sandbox-pid1.h"
#include "src/main/tools/linux-sandbox-utils.h"
#include "src/main/tools/process-tools.h"

int global_outer_uid;
int global_outer_gid;

// The PID of our child.
static volatile sig_atomic_t global_child_pid;

// The signal that will be sent to the child when a timeout occurs.
static volatile sig_atomic_t global_next_timeout_signal = SIGTERM;

// Whether the child was killed due to a timeout.
static volatile sig_atomic_t global_timeout_occurred;

static void CloseFds() {
  DIR *fds = opendir("/proc/self/fd");
  if (fds == nullptr) {
    DIE("opendir");
  }

  while (1) {
    errno = 0;
    struct dirent *dent = readdir(fds);

    if (dent == nullptr) {
      if (errno != 0) {
        DIE("readdir");
      }
      break;
    }

    if (isdigit(dent->d_name[0])) {
      errno = 0;
      int fd = strtol(dent->d_name, nullptr, 10);

      // (1) Skip unparseable entries.
      // (2) Close everything except stdin, stdout and stderr.
      // (3) Do not accidentally close our directory handle.
      if (errno == 0 && fd > STDERR_FILENO && fd != dirfd(fds)) {
        if (close(fd) < 0) {
          DIE("close");
        }
      }
    }
  }

  if (closedir(fds) < 0) {
    DIE("closedir");
  }
}

static void OnTimeout(int sig) {
  global_timeout_occurred = true;
  kill(global_child_pid, global_next_timeout_signal);
  if (global_next_timeout_signal == SIGTERM && opt.kill_delay_secs > 0) {
    global_next_timeout_signal = SIGKILL;
    SetTimeout(opt.kill_delay_secs);
  }
}

static void ForwardSignal(int signum) {
  if (global_child_pid > 0) {
    kill(global_child_pid, signum);
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
      // It's fine to use the default handler for SIGCHLD, because we use
      // waitpid() in the main loop to wait for our child to die anyway.
      case SIGCHLD:
        break;
      // One does not simply install a signal handler for these two signals
      case SIGKILL:
      case SIGSTOP:
        break;
      // Ignore SIGTTIN and SIGTTOU, as we hand off the terminal to the child in
      // SpawnChild() later.
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

static int SpawnPid1() {
  const int kStackSize = 1024 * 1024;
  std::vector<char> child_stack(kStackSize);

  int sync_pipe[2];
  if (pipe(sync_pipe) < 0) {
    DIE("pipe");
  }

  int clone_flags =
      CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWIPC | CLONE_NEWPID | SIGCHLD;
  if (opt.create_netns) {
    clone_flags |= CLONE_NEWNET;
  }
  if (opt.fake_hostname) {
    clone_flags |= CLONE_NEWUTS;
  }

  // We use clone instead of unshare, because unshare sometimes fails with
  // EINVAL due to a race condition in the Linux kernel (see
  // https://lkml.org/lkml/2015/7/28/833).
  int child_pid =
      clone(Pid1Main, child_stack.data() + kStackSize, clone_flags, sync_pipe);
  if (child_pid < 0) {
    DIE("clone");
  }

  PRINT_DEBUG("linux-sandbox-pid1 has PID %d", child_pid);

  // We close the write end of the sync pipe, read a byte and then close the
  // pipe. This proves to the linux-sandbox-pid1 process that we still existed
  // after it ran prctl(PR_SET_PDEATHSIG, SIGKILL), thus preventing a race
  // condition where the parent is killed before that call was made.
  char buf;
  if (close(sync_pipe[1]) < 0) {
    DIE("close");
  }
  if (read(sync_pipe[0], &buf, 1) < 0) {
    DIE("read");
  }
  if (close(sync_pipe[0]) < 0) {
    DIE("close");
  }

  return child_pid;
}

static int WaitForPid1(int child_pid) {
  int err, status;
  do {
    err = waitpid(child_pid, &status, 0);
  } while (err < 0 && errno == EINTR);

  if (err < 0) {
    DIE("waitpid");
  }

  if (global_timeout_occurred) {
    // The child exited because we killed it due to receiving a signal
    // ourselves. Do not trust the exitcode in this case, just calculate it from
    // the signal.
    PRINT_DEBUG("child exited due to timeout");
    return 128 + SIGALRM;
  } else if (WIFSIGNALED(status)) {
    PRINT_DEBUG("child exited due to receiving signal: %s",
                strsignal(WTERMSIG(status)));
    return 128 + WTERMSIG(status);
  } else {
    PRINT_DEBUG("child exited normally with exitcode %d", WEXITSTATUS(status));
    return WEXITSTATUS(status);
  }
}

int main(int argc, char *argv[]) {
  KillMeWhenMyParentDies(SIGKILL);
  DropPrivileges();
  ParseOptions(argc, argv);

  Redirect(opt.stdout_path, STDOUT_FILENO);
  Redirect(opt.stderr_path, STDERR_FILENO);

  global_outer_uid = getuid();
  global_outer_gid = getgid();

  // Make sure the sandboxed process does not inherit any accidentally left open
  // file handles from our parent.
  CloseFds();

  SetupSignalHandlers();
  global_child_pid = SpawnPid1();

  if (opt.timeout_secs > 0) {
    SetTimeout(opt.timeout_secs);
  }

  return WaitForPid1(global_child_pid);
}
