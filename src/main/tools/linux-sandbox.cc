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

#include "linux-sandbox-options.h"
#include "linux-sandbox-pid1.h"
#include "linux-sandbox-utils.h"

#define DIE(args...)                                     \
  {                                                      \
    fprintf(stderr, __FILE__ ":" S__LINE__ ": \"" args); \
    fprintf(stderr, "\": ");                             \
    perror(NULL);                                        \
    exit(EXIT_FAILURE);                                  \
  }

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

int global_outer_uid;
int global_outer_gid;

static int global_child_pid;

// The signal that will be sent to the child when a timeout occurs.
static volatile sig_atomic_t global_next_timeout_signal = SIGTERM;

// The signal that caused us to kill the child (e.g. on timeout).
static volatile sig_atomic_t global_signal;

static void CloseFds() {
  DIR *fds = opendir("/proc/self/fd");
  if (fds == NULL) {
    DIE("opendir");
  }

  while (1) {
    errno = 0;
    struct dirent *dent = readdir(fds);

    if (dent == NULL) {
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

static void HandleSignal(int signum, void (*handler)(int)) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handler;
  if (sigemptyset(&sa.sa_mask) < 0) {
    DIE("sigemptyset");
  }
  if (sigaction(signum, &sa, NULL) < 0) {
    DIE("sigaction");
  }
}

static void OnTimeout(int sig) {
  global_signal = sig;
  kill(global_child_pid, global_next_timeout_signal);
  if (global_next_timeout_signal == SIGTERM && opt.kill_delay_secs > 0) {
    global_next_timeout_signal = SIGKILL;
    alarm(opt.kill_delay_secs);
  }
}

static void SpawnPid1() {
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
  global_child_pid =
      clone(Pid1Main, child_stack.data() + kStackSize, clone_flags, sync_pipe);
  if (global_child_pid < 0) {
    DIE("clone");
  }

  PRINT_DEBUG("linux-sandbox-pid1 has PID %d", global_child_pid);

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
}

static int WaitForPid1() {
  int err, status;
  do {
    err = waitpid(global_child_pid, &status, 0);
  } while (err < 0 && errno == EINTR);

  if (err < 0) {
    DIE("waitpid");
  }

  if (global_signal > 0) {
    // The child exited because we killed it due to receiving a signal
    // ourselves. Do not trust the exitcode in this case, just calculate it from
    // the signal.
    PRINT_DEBUG("child exited due to us catching signal: %s",
                strsignal(global_signal));
    return 128 + global_signal;
  } else if (WIFSIGNALED(status)) {
    PRINT_DEBUG("child exited due to receiving signal: %s",
                strsignal(WTERMSIG(status)));
    return 128 + WTERMSIG(status);
  } else {
    PRINT_DEBUG("child exited normally with exitcode %d", WEXITSTATUS(status));
    return WEXITSTATUS(status);
  }
}

static void Redirect(const std::string &target_path, int fd) {
  if (!target_path.empty() && target_path != "-") {
    const int flags = O_WRONLY | O_CREAT | O_TRUNC | O_APPEND;
    int fd_out = open(target_path.c_str(), flags, 0666);
    if (fd_out < 0) {
      DIE("open(%s)", target_path.c_str());
    }
    // If we were launched with less than 3 fds (stdin, stdout, stderr) open,
    // but redirection is still requested via a command-line flag, something is
    // wacky and the following code would not do what we intend to do, so let's
    // bail.
    if (fd_out < 3) {
      DIE("open(%s) returned a handle that is reserved for stdin / stdout / "
          "stderr",
          target_path.c_str());
    }
    if (dup2(fd_out, fd) < 0) {
      DIE("dup2()");
    }
    if (close(fd_out) < 0) {
      DIE("close()");
    }
  }
}

int main(int argc, char *argv[]) {
  // Ask the kernel to kill us with SIGKILL if our parent dies.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0) {
    DIE("prctl");
  }

  ParseOptions(argc, argv);

  Redirect(opt.stdout_path, STDOUT_FILENO);
  Redirect(opt.stderr_path, STDERR_FILENO);

  // This should never be called as a setuid binary, drop privileges just in
  // case. We don't need to be root, because we use user namespaces anyway.
  if (setuid(getuid()) < 0) {
    DIE("setuid");
  }

  global_outer_uid = getuid();
  global_outer_gid = getgid();

  // Make sure the sandboxed process does not inherit any accidentally left open
  // file handles from our parent.
  CloseFds();

  HandleSignal(SIGALRM, OnTimeout);
  if (opt.timeout_secs > 0) {
    alarm(opt.timeout_secs);
  }

  SpawnPid1();
  return WaitForPid1();
}
