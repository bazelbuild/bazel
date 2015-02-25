#define _GNU_SOURCE

// Copyright 2014 Google Inc. All rights reserved.
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
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <linux/capability.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int global_debug = 0;
static int global_cpid; // Returned by fork()

#define PRINT_DEBUG(...) do { if (global_debug) {fprintf(stderr, "sandbox.c: " __VA_ARGS__);}} while(0)

#define CHECK_CALL(x) if ((x) == -1) { perror(#x); exit(1); }
#define CHECK_NOT_NULL(x) if (x == NULL) { perror(#x); exit(1); }
#define DIE() do { fprintf(stderr, "Error in %d\n", __LINE__); exit(-1); } while(0);

int kChildrenCleanupDelay = 1;

void Usage() {
  fprintf(stderr,
          "Usage: ./sandbox [-R sandbox-root] [-m mount] -C command arg1\n"
          "Mandatory arguments:\n"
          "  -C command to run inside sandbox, followed by arguments\n"
          "  -S directory which will become the root of the sandbox\n"
          "\n"
          "Optional arguments:\n"
          "  -t absolute path to bazel tools directory\n"
          "  -T timeout after which sandbox will be terminated\n"
          "  -m system directory to mount inside the sandbox\n"
          " Multiple directories can be specified and each of them will\n"
          " be mount as readonly\n"
          "  -D if set, debug info will be printed\n");
  exit(1);
}

void PropagateSignals();
void EnableAlarm();
void SetupSlashDev();

static volatile sig_atomic_t global_signal_received = 0;

int main(int argc, char *argv[]) {
  char *include_prefix = NULL;
  char *sandbox_root = NULL;
  char *tools = NULL;
  char **mounts = malloc(argc * sizeof(char*));
  char **includes = malloc(argc * sizeof(char*));
  int num_mounts = 0;
  int num_includes = 0;
  int iArg = 0;
  int uid = getuid();
  int gid = getgid();
  int timeout = 0;

  for (iArg = 1; iArg < argc - 1; iArg++) {
    if (strlen(argv[iArg]) != 2) {
      Usage();
    }
    if (argv[iArg][0] != '-') {
      Usage();
    }
    switch (argv[iArg][1]) {
      case 'S':
        if (sandbox_root == NULL) {
          sandbox_root = argv[++iArg];
        } else {
          fprintf(stderr,
                  "Multiple sandbox roots (-S) specified (expected one).\n");
          Usage();
        }
        break;
      case 'm':
        mounts[num_mounts++] = argv[++iArg];
        break;
      case 'D':
        global_debug = 1;
        break;
      case 'T':
        sscanf(argv[iArg], "%d", &timeout);
        break;
      case 'N':
        include_prefix = argv[++iArg];
        break;
      case 'n':
        includes[num_includes++] = argv[++iArg];
        break;
      case 'C':
        iArg++;
        goto parsing_finished;
      case 't':
        tools = argv[++iArg];
        break;
      default:
        fprintf(stderr, "Unrecognized argument: %s\n", argv[iArg]);
        Usage();
    }
  }

parsing_finished:
  if (iArg == argc) {
    fprintf(stderr, "No command specified.\n");
    Usage();
  }
  if (timeout < 0) {
    fprintf(stderr, "Invalid timeout (-T) value: %d", timeout);
    Usage();
  }

  // parsed all arguments, now prepare sandbox

  PRINT_DEBUG("%s\n", sandbox_root);
  // create new namespaces in which this process and its children will live
  CHECK_CALL(unshare(CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWUSER));
  CHECK_CALL(mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL));
  // mount sandbox and go there
  CHECK_CALL(mount(sandbox_root, sandbox_root, NULL, MS_BIND | MS_NOSUID, NULL));
  CHECK_CALL(chdir(sandbox_root));

  SetupSlashDev();
  // mount blaze specific directories - tools/ and build-runfiles/
  if (tools != NULL) {
    PRINT_DEBUG("tools: %s\n", tools);
    CHECK_CALL(mkdir("tools", 0755));
    CHECK_CALL(mount(tools, "tools", NULL, MS_BIND | MS_RDONLY, NULL));
  }

  // mounts passed in argv; those are mostly dirs for shared libs
  for (int i = 0; i < num_mounts; i++) {
    CHECK_CALL(mount(mounts[i], mounts[i] + 1, NULL, MS_BIND | MS_RDONLY, NULL));
  }

  // c++ compilation
  // headers go in separate directory
  if (include_prefix != NULL) {
    CHECK_CALL(chdir(include_prefix));
    for (int i = 0; i < num_includes; i++) {
      // TODO(bazel-team) sometimes list of -iquote given by bazel contains
      // invalid (non-existing) entries, ideally we would like not to have them
      PRINT_DEBUG("include: %s\n", includes[i]);
      if (mount(includes[i], includes[i] + 1 , NULL, MS_BIND, NULL) > -1) {
        continue;
      }
      if (errno == ENOENT) {
        continue;
      }
      CHECK_CALL(-1);
    }
    CHECK_CALL(chdir(".."));
  }

  // set group and user mapping from outer namespace to inner:
  // no changes in the parent, be root in the child
  int uid_fd, gid_fd;
  char uid_mapping[64], gid_mapping[64];
  sprintf(uid_mapping, "0 %d 1\n", uid);
  sprintf(gid_mapping, "0 %d 1\n", gid);
  uid_fd = open("/proc/self/uid_map", O_WRONLY);
  CHECK_CALL(uid_fd);
  CHECK_CALL(write(uid_fd, uid_mapping, strlen(uid_mapping)));
  CHECK_CALL(close(uid_fd));

  gid_fd = open("/proc/self/gid_map", O_WRONLY);
  CHECK_CALL(gid_fd);
  CHECK_CALL(write(gid_fd, gid_mapping, strlen(gid_mapping)));
  CHECK_CALL(close(gid_fd));

  CHECK_CALL(setresuid(0, 0, 0));
  CHECK_CALL(setresgid(0, 0, 0));

  CHECK_CALL(mkdir("proc", 0755));
  CHECK_CALL(mount("/proc", "proc", NULL, MS_REC | MS_BIND, NULL));
  // make sandbox actually hermetic:
  // move the real root to old_root, then detach it
  char old_root[16] = "old-root-XXXXXX";
  CHECK_NOT_NULL(mkdtemp(old_root));
  // pivot_root has no wrapper in libc, so we need syscall()
  CHECK_CALL(syscall(SYS_pivot_root, ".", old_root));
  CHECK_CALL(chroot("."));
  CHECK_CALL(umount2(old_root, MNT_DETACH));
  CHECK_CALL(rmdir(old_root));

  free(mounts);
  free(includes);

  for (int i = iArg; i < argc; i += 1) {
    PRINT_DEBUG("arg: %s\n", argv[i]);
  }

  // spawn child and wait until it finishes
  global_cpid = fork();
  if (global_cpid == 0) {
    CHECK_CALL(setpgid(0, 0));
    // if the execvp below fails with "No such file or directory" it means that:
    // a) the binary is not in the sandbox (which means it wasn't included in
    // the inputs)
    // b) the binary uses shared library which is not inside sandbox - you can
    // check for that by running "ldd ./a.out" (by default directories
    // starting with /lib* and /usr/lib* should be there)
    // c) the binary uses elf interpreter which is not inside sandbox - you can
    // check for that by running "readelf -a a.out | grep interpreter" (the
    // sandbox code assumes that it is either in /lib*/ or /usr/lib*/)
    CHECK_CALL(execvp(argv[iArg], argv + iArg));
    PRINT_DEBUG("Exec failed near %s:%d\n", __FILE__, __LINE__);
    exit(1);
  } else {
    // PARENT
    // make sure that all signals propagate to children (mostly useful to kill
    // entire sandbox)
    PropagateSignals();
    // after given timeout, kill children
    EnableAlarm(timeout);
    int status = 0;
    while (1) {
      PRINT_DEBUG("Waiting for the child...\n");
      pid_t pid = wait(&status);
      if (global_signal_received) {
        PRINT_DEBUG("Received signal: %s\n", strsignal(global_signal_received));
        CHECK_CALL(killpg(global_cpid, global_signal_received));
        // give children some time for cleanup before they terminate
        sleep(kChildrenCleanupDelay);
        CHECK_CALL(killpg(global_cpid, SIGKILL));
        exit(128 | global_signal_received);
      }
      if (errno == EINTR) {
        continue;
      }
      if (pid < 0) {
        perror("Wait failed:");
        exit(1);
      }
      if (WIFEXITED(status)) {
        PRINT_DEBUG("Child exited with status: %d\n", WEXITSTATUS(status));
        exit(WEXITSTATUS(status));
      }
      if (WIFSIGNALED(status)) {
        PRINT_DEBUG("Child terminated by a signal: %d\n", WTERMSIG(status));
        exit(WEXITSTATUS(status));
      }
      if (WIFSTOPPED(status)) {
        PRINT_DEBUG("Child stopped by a signal: %d\n", WSTOPSIG(status));
      }
    }
  }

  return 0;
}

void SignalHandler(int signum, siginfo_t *info, void *uctxt) {
  global_signal_received = signum;
}

void PropagateSignals() {
  // propagate some signals received by the parent to processes in sandbox, so
  // that it's easier to terminate entire sandbox
  struct sigaction action = {};
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = SignalHandler;

  // handle all signals that could terminate the process
  int signals[] = {SIGHUP, SIGINT, SIGKILL, SIGPIPE, SIGALRM, SIGTERM, SIGPOLL,
    SIGPROF, SIGVTALRM,
    // signals below produce core dump by default, however at the moment we'll
    // just terminate
    SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGBUS, SIGSYS, SIGTRAP, SIGXCPU,
    SIGXFSZ, -1};
  for (int *p = signals; *p != -1; p++) {
    sigaction(*p, &action, NULL);
  }
}

void SetupSlashDev() {
  CHECK_CALL(mkdir("dev", 0755));
  const char *devs[] = {
    "/dev/null",
    "/dev/random",
    "/dev/urandom",
    "/dev/zero",
    NULL
  };
  for (int i = 0; devs[i] != NULL; i++) {
    // open+close to create the file, which will become mount point for actual
    // device
    int handle = open(devs[i] + 1, O_CREAT | O_RDONLY, 0644);
    CHECK_CALL(handle);
    CHECK_CALL(close(handle));
    CHECK_CALL(mount(devs[i], devs[i] + 1, NULL, MS_BIND, NULL));
  }
}

void EnableAlarm(int timeout) {
  if (timeout <= 0) return;

  struct itimerval timer = {};
  timer.it_value.tv_sec = (long) timeout;
  CHECK_CALL(setitimer(ITIMER_REAL, &timer, NULL));
}
