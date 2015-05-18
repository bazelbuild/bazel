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
#include <stdarg.h>
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

#define PRINT_DEBUG(...) do { if (global_debug) {fprintf(stderr, "sandbox.c: " __VA_ARGS__);}} while(0)

#define CHECK_CALL(x) if ((x) == -1) { perror(#x); exit(1); }
#define CHECK_NOT_NULL(x) if (x == NULL) { perror(#x); exit(1); }
#define DIE() do { fprintf(stderr, "Error in %d\n", __LINE__); exit(-1); } while(0);

const int kChildrenCleanupDelay = 1;

static volatile sig_atomic_t global_signal_received = 0;

//
// Options parsing result
//
struct Options {
  char **args;          // Command to run (-C / --)
  char *include_prefix; // Include prefix (-N)
  char *sandbox_root;   // Sandbox root (-S)
  char *tools;          // tools directory (-t)
  char **mounts;        // List of directories to mount (-m)
  char **includes;      // List of include directories (-n)
  int num_mounts;       // size of mounts
  int num_includes;     // size of includes
  int timeout;          // Timeout (-T)
};

// Print out a usage error. argc and argv are the argument counter
// and vector, fmt is a format string for the error message to print.
void Usage(int argc, char **argv, char *fmt, ...);
// Parse the command line flags and return the result in an
// Options structure passed as argument.
void ParseCommandLine(int argc, char **argv, struct Options *opt);

// Signal hanlding
void PropagateSignals();
void EnableAlarm();
// Sandbox setup
void SetupDirectories(struct Options* opt);
void SetupSlashDev();
void SetupUserNamespace(int uid, int gid);
void ChangeRoot();
// Write the file "filename" using a format string specified by "fmt".
// Returns -1 on failure.
int WriteFile(const char *filename, const char *fmt, ...);
// Run the command specified by the argv array and kill it after
// timeout seconds.
void SpawnCommand(char **argv, int timeout);



int main(int argc, char *argv[]) {
  struct Options opt = {
    .args = NULL,
    .include_prefix = NULL,
    .sandbox_root = NULL,
    .tools = NULL,
    .mounts = calloc(argc, sizeof(char*)),
    .includes = calloc(argc, sizeof(char*)),
    .num_mounts = 0,
    .num_includes = 0,
    .timeout = 0
  };
  ParseCommandLine(argc, argv, &opt);
  int uid = getuid();
  int gid = getgid();

  // parsed all arguments, now prepare sandbox
  PRINT_DEBUG("%s\n", opt.sandbox_root);
  // create new namespaces in which this process and its children will live
  CHECK_CALL(unshare(CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWUSER));
  CHECK_CALL(mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL));
  // Create the sandbox directory layout
  SetupDirectories(&opt);
  // Set the user namespace (user_namespaces(7))
  SetupUserNamespace(uid, gid);
  // make sandbox actually hermetic:
  ChangeRoot();

  // Finally call the command
  free(opt.mounts);
  free(opt.includes);
  SpawnCommand(opt.args, opt.timeout);
  return 0;
}

void SpawnCommand(char **argv, int timeout) {
  for (int i = 0; argv[i] != NULL; i++) {
    PRINT_DEBUG("arg: %s\n", argv[i]);
  }

  // spawn child and wait until it finishes
  pid_t cpid = fork();
  if (cpid == 0) {
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
    CHECK_CALL(execvp(argv[0], argv));
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
        CHECK_CALL(killpg(cpid, global_signal_received));
        // give children some time for cleanup before they terminate
        sleep(kChildrenCleanupDelay);
        CHECK_CALL(killpg(cpid, SIGKILL));
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
}

int WriteFile(const char *filename, const char *fmt, ...) {
  int r;
  va_list ap;
  FILE *stream = fopen(filename, "w");
  if (stream == NULL) {
    return -1;
  }
  va_start(ap, fmt);
  r = vfprintf(stream, fmt, ap);
  va_end(ap);
  if (r >= 0) {
    r = fclose(stream);
  }
  return r;
}

//
// Signal handling
//
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

void EnableAlarm(int timeout) {
  if (timeout <= 0) return;

  struct itimerval timer = {};
  timer.it_value.tv_sec = (long) timeout;
  CHECK_CALL(setitimer(ITIMER_REAL, &timer, NULL));
}

//
// Sandbox setup
//
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

void SetupDirectories(struct Options *opt) {
  // Mount the sandbox and go there.
  CHECK_CALL(mount(opt->sandbox_root, opt->sandbox_root, NULL, MS_BIND | MS_NOSUID, NULL));
  CHECK_CALL(chdir(opt->sandbox_root));
  SetupSlashDev();
  // Mount blaze specific directories - tools/ and build-runfiles/.
  if (opt->tools != NULL) {
    PRINT_DEBUG("tools: %s\n", opt->tools);
    CHECK_CALL(mkdir("tools", 0755));
    CHECK_CALL(mount(opt->tools, "tools", NULL, MS_BIND | MS_RDONLY, NULL));
  }

  // Mount directories passed in argv; those are mostly dirs for shared libs.
  for (int i = 0; i < opt->num_mounts; i++) {
    CHECK_CALL(mount(opt->mounts[i], opt->mounts[i] + 1, NULL, MS_BIND | MS_RDONLY, NULL));
  }

  // C++ compilation
  // C++ headers go in a separate directory.
  if (opt->include_prefix != NULL) {
    CHECK_CALL(chdir(opt->include_prefix));
    for (int i = 0; i < opt->num_includes; i++) {
      // TODO(bazel-team): sometimes list of -iquote given by bazel contains
      // invalid (non-existing) entries, ideally we would like not to have them
      PRINT_DEBUG("include: %s\n", opt->includes[i]);
      if (mount(opt->includes[i], opt->includes[i] + 1 , NULL, MS_BIND, NULL) > -1) {
        continue;
      }
      if (errno == ENOENT) {
        continue;
      }
      CHECK_CALL(-1);
    }
    CHECK_CALL(chdir(".."));
  }

  CHECK_CALL(mkdir("proc", 0755));
  CHECK_CALL(mount("/proc", "proc", NULL, MS_REC | MS_BIND, NULL));
}

void SetupUserNamespace(int uid, int gid) {
  // Disable needs for CAP_SETGID
  int r = WriteFile("/proc/self/setgroups", "deny");
  if (r < 0 && errno != ENOENT) {
    // Writing to /proc/self/setgroups might fail on earlier
    // version of linux because setgroups does not exist, ignore.
    perror("WriteFile(\"/proc/self/setgroups\", \"deny\")");
    exit(-1);
  }
  // set group and user mapping from outer namespace to inner:
  // no changes in the parent, be root in the child
  CHECK_CALL(WriteFile("/proc/self/uid_map", "0 %d 1\n", uid));
  CHECK_CALL(WriteFile("/proc/self/gid_map", "0 %d 1\n", gid));

  CHECK_CALL(setresuid(0, 0, 0));
  CHECK_CALL(setresgid(0, 0, 0));
}

void ChangeRoot() {
  // move the real root to old_root, then detach it
  char old_root[16] = "old-root-XXXXXX";
  CHECK_NOT_NULL(mkdtemp(old_root));
  // pivot_root has no wrapper in libc, so we need syscall()
  CHECK_CALL(syscall(SYS_pivot_root, ".", old_root));
  CHECK_CALL(chroot("."));
  CHECK_CALL(umount2(old_root, MNT_DETACH));
  CHECK_CALL(rmdir(old_root));
}

//
// Command line parsing
//
void Usage(int argc, char **argv, char *fmt, ...) {
  int i;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr,
          "\nUsage: %s [-S sandbox-root] [-m mount] [-C|--] command arg1\n",
          argv[0]);
  fprintf(stderr, "  provided:");
  for (i = 0; i < argc; i++) {
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(stderr,
          "\nMandatory arguments:\n"
          "  [-C|--] command to run inside sandbox, followed by arguments\n"
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

void ParseCommandLine(int argc, char **argv, struct Options *opt) {
  extern char *optarg;
  extern int optind, optopt;
  int c;

  opt->include_prefix = NULL;
  opt->sandbox_root = NULL;
  opt->tools = NULL;
  opt->mounts = malloc(argc * sizeof(char*));
  opt->includes = malloc(argc * sizeof(char*));
  opt->num_mounts = 0;
  opt->num_includes = 0;
  opt->timeout = 0;

  while ((c = getopt(argc, argv, "+:S:t:T:m:N:n:DC")) != -1) {
    switch(c) {
      case 'S':
        if (opt->sandbox_root == NULL) {
          opt->sandbox_root = optarg;
        } else {
          Usage(argc, argv,
                "Multiple sandbox roots (-S) specified (expected one).");
        }
        break;
      case 'm':
        opt->mounts[opt->num_mounts++] = optarg;
        break;
      case 'D':
        global_debug = 1;
        break;
      case 'T':
        sscanf(optarg, "%d", &opt->timeout);
        if (opt->timeout < 0) {
          Usage(argc, argv, "Invalid timeout (-T) value: %d", opt->timeout);
        }
        break;
      case 'N':
        opt->include_prefix = optarg;
        break;
      case 'n':
        opt->includes[opt->num_includes++] = optarg;
        break;
      case 'C':
        break; // deprecated, ignore.
      case 't':
        opt->tools = optarg;
        break;
      case '?':
        Usage(argc, argv, "Unrecognized argument: -%c (%d)", optopt, optind);
        break;
      case ':':
        Usage(argc, argv, "Flag -%c requires an argument", optopt);
        break;
    }
  }

  opt->args = argv + optind;
  if (argc <= optind) {
    Usage(argc, argv, "No command specified");
  }
}
