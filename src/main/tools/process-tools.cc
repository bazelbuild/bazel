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

#include "src/main/tools/process-tools.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#if defined(__FreeBSD__)
#include <sys/procctl.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

using std::vector;

// Drops privileges irrevocably to the real uid / gid by setting the effective
// and saved uid / gid to the real uid / gid. Useful if we happen to have been
// called as a setuid-/setgid-root binary.
void DropPrivileges() {
  if (setgid(getgid()) < 0) {
    DIE("setgid");
  }
  if (setuid(getuid()) < 0) {
    DIE("setuid");
  }
}

void Redirect(const std::string &target_path, int fd) {
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

void WriteFile(const std::string &filename, const char *fmt, ...) {
  FILE *stream = fopen(filename.c_str(), "w");
  if (stream == nullptr) {
    DIE("fopen(%s)", filename.c_str());
  }

  va_list ap;
  va_start(ap, fmt);
  // Use a local variable to make sure we call va_end before DIE() in case this
  // returns an error.
  int r = vfprintf(stream, fmt, ap);
  va_end(ap);

  if (r < 0) {
    DIE("vfprintf");
  }

  if (fclose(stream) != 0) {
    DIE("fclose(%s)", filename.c_str());
  }
}

void SetTimeout(double timeout_secs) {
  if (timeout_secs <= 0) {
    DIE("timeout_secs must be positive");
  }

  double int_val, fraction_val;
  fraction_val = modf(timeout_secs, &int_val);

  struct itimerval timer;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0;
  timer.it_value.tv_sec = (time_t)int_val,
  timer.it_value.tv_usec = (suseconds_t)(fraction_val * 1e6);

  if (setitimer(ITIMER_REAL, &timer, nullptr) < 0) {
    DIE("setitimer");
  }
}

void InstallSignalHandler(int signum, void (*handler)(int)) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handler;
  if (handler == SIG_IGN || handler == SIG_DFL) {
    // No point in blocking signals when using the default handler or ignoring
    // the signal.
    if (sigemptyset(&sa.sa_mask) < 0) {
      DIE("sigemptyset");
    }
  } else {
    // When using a custom handler, block all signals from firing while the
    // handler is running.
    if (sigfillset(&sa.sa_mask) < 0) {
      DIE("sigfillset");
    }
  }
  // sigaction may fail for certain reserved signals. Ignore failure in this
  // case.
  sigaction(signum, &sa, nullptr);
}

void IgnoreSignal(int signum) { InstallSignalHandler(signum, SIG_IGN); }

void RestoreSignalHandlersAndMask() {
  // Use an empty signal mask for the process (= unblock all signals).
  sigset_t empty_set;
  if (sigemptyset(&empty_set) < 0) {
    DIE("sigemptyset");
  }
  if (sigprocmask(SIG_SETMASK, &empty_set, nullptr) < 0) {
    DIE("sigprocmask(SIG_SETMASK, <empty set>, nullptr)");
  }

  // Set the default signal handler for all signals.
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  if (sigemptyset(&sa.sa_mask) < 0) {
    DIE("sigemptyset");
  }
  sa.sa_handler = SIG_DFL;
  for (int i = 1; i < NSIG; ++i) {
    // Ignore possible errors, because we might not be allowed to set the
    // handler for certain signals, but we still want to try.
    sigaction(i, &sa, nullptr);
  }
}

void KillMeWhenMyParentDies(int signum) {
#if defined(__linux__)
  if (prctl(PR_SET_PDEATHSIG, signum) < 0) {
    DIE("prctl");
  }
#endif
}

void BecomeSubreaper() {
#if defined(__FreeBSD__)
  if (procctl(P_PID, getpid(), PROC_REAP_ACQUIRE, 0) < 0) {
    DIE("procctl");
  }
#endif
#if defined(__linux__)
  if (prctl(PR_SET_CHILD_SUBREAPER, 1) < 0) {
    DIE("prctl");
  }
#endif
}

int SpawnCommand(const vector<char *> &args) {
  int child_pid = fork();
  if (child_pid < 0) {
    DIE("fork");
  } else if (child_pid == 0) {
    // Put the child into its own process group.
    if (setpgid(0, 0) < 0) {
      DIE("setpgid");
    }

    // Try to assign our terminal to the child process.
    if (tcsetpgrp(STDIN_FILENO, getpgrp()) < 0 && errno != ENOTTY) {
      DIE("tcsetpgrp")
    }

    // Unblock all signals, restore default handlers.
    RestoreSignalHandlersAndMask();

    // Force umask to include read and execute for everyone, to make output
    // permissions predictable.
    umask(022);

    if (execvp(args[0], args.data()) < 0) {
      DIE("execvp(%s, %p)", args[0], args.data());
    }
  }
  return child_pid;
}

static void KillAllRemainingChildren(int main_child_pid) {
  // If the child process we spawned earlier terminated, we want to make
  // sure all remaining (grand)children are killed, too.
  if (getpid() == 1) {
    // If we're PID 1, this is easy.
    if (kill(-1, SIGKILL) < 0 && errno != ESRCH) {
      DIE("kill");
    }
  } else {
#if defined(__FreeBSD__)
    // FreeBSD is cool, because it has an API to kill all our descendants in one
    // go.
    struct procctl_reaper_kill data;
    data.rk_sig = SIGKILL;
    if (procctl(P_PID, getpid(), PROC_REAP_KILL, &data) < 0 && errno != ESRCH) {
      DIE("procctl")
    }
#else
    // On other operating systems, we have to resort to sending SIGKILL to the
    // process group of our child and hope that this kills them all.
    // TODO(philwo) - what if a child switched to a different process group
    // and we can't kill it like this? Maybe parse /proc/*/stat and filter
    // by "their PPID = my PID"?
    if (kill(-main_child_pid, SIGKILL) < 0 && errno != ESRCH) {
      DIE("kill");
    }
#endif
  }
}

int WaitForChild(int main_child_pid) {
  // This will be overwritten by the real exitcode from the child in the loop
  // below. In case something goes horribly wrong and that doesn't happen, at
  // least exit with a failure.
  int exitcode = EXIT_FAILURE;
  while (1) {
    // Check for zombies to be reaped and exit, if our own child exited.
    int status;
    pid_t killed_pid = wait(&status);

    if (killed_pid < 0) {
      // Our PID1 process got a signal that interrupted the wait() call and that
      // was either ignored or forwarded to the child. This is expected and
      // fine, just continue waiting.
      if (errno == EINTR) {
        continue;
      } else if (errno == ECHILD) {
        // No children left to wait for, we're done here.
        break;
      }
      DIE("waitpid")
    } else {
      if (killed_pid == main_child_pid) {
        KillAllRemainingChildren(main_child_pid);

        if (WIFSIGNALED(status)) {
          exitcode = 128 + WTERMSIG(status);
        } else {
          exitcode = WEXITSTATUS(status);
        }
      }
    }
  }
  return exitcode;
}
