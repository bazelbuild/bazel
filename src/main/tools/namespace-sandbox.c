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

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <pwd.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "process-tools.h"

#define PRINT_DEBUG(...)                                        \
  do {                                                          \
    if (global_debug) {                                         \
      fprintf(stderr, __FILE__ ":" S__LINE__ ": " __VA_ARGS__); \
    }                                                           \
  } while (0)

static bool global_debug = false;
static double global_kill_delay;
static int global_child_pid;
static volatile sig_atomic_t global_signal;

// The uid and gid of the user and group 'nobody'.
static const int kNobodyUid = 65534;
static const int kNobodyGid = 65534;

// Options parsing result.
struct Options {
  double timeout_secs;     // How long to wait before killing the child (-T)
  double kill_delay_secs;  // How long to wait before sending SIGKILL in case of
                           // timeout (-t)
  const char *stdout_path;   // Where to redirect stdout (-l)
  const char *stderr_path;   // Where to redirect stderr (-L)
  char *const *args;         // Command to run (--)
  const char *sandbox_root;  // Sandbox root (-S)
  const char *working_dir;   // Working directory (-W)
  char **mount_sources;      // Map of directories to mount, from
  char **mount_targets;      // sources -> targets (-m)
  int num_mounts;            // How many mounts were specified
};

// Child function used by CheckNamespacesSupported() in call to clone().
static int CheckNamespacesSupportedChild(void *arg) { return 0; }

// Check whether the required namespaces are supported.
static int CheckNamespacesSupported() {
  const int stackSize = 1024 * 1024;
  char *stack;
  char *stackTop;
  pid_t pid;

  // Allocate stack for child.
  stack = malloc(stackSize);
  if (stack == NULL) {
    DIE("malloc failed\n");
  }

  // Assume stack grows downward.
  stackTop = stack + stackSize;

  // Create child with own namespaces. We use clone() instead of unshare() here
  // because of the kernel bug (ref. CreateNamespaces) that lets unshare fail
  // sometimes. As this check has to run as fast as possible, we can't afford to
  // spend time sleeping and retrying here until it eventually works (or not).
  CHECK_CALL(pid = clone(CheckNamespacesSupportedChild, stackTop,
                         CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWUTS |
                             CLONE_NEWIPC | SIGCHLD,
                         NULL));
  CHECK_CALL(waitpid(pid, NULL, 0));

  return EXIT_SUCCESS;
}

// Print out a usage error. argc and argv are the argument counter and vector,
// fmt is a format,
// string for the error message to print.
static void Usage(int argc, char *const *argv, const char *fmt, ...) {
  int i;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr,
          "\nUsage: %s [-S sandbox-root] [-W working-dir] [-M source -m "
          "target] -- command arg1\n",
          argv[0]);
  fprintf(stderr, "  provided:");
  for (i = 0; i < argc; i++) {
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(stderr,
          "\nMandatory arguments:\n"
          "  -S directory which will become the root of the sandbox\n"
          "  -- command to run inside sandbox, followed by arguments\n"
          "\n"
          "Optional arguments:\n"
          "  -W working directory\n"
          "  -t time to give the child to shutdown cleanly before sending it a "
          "SIGKILL\n"
          "  -T timeout after which sandbox will be terminated\n"
          "  -t in case timeout occurs, how long to wait before killing the "
          "child with SIGKILL\n"
          "  -M/-m system directory to mount inside the sandbox\n"
          "    Multiple directories can be specified and each of them will\n"
          "    be mounted readonly. The -M option specifies which directory\n"
          "    to mount, the -m option specifies where to mount it in the\n"
          "    sandbox.\n"
          "  -D if set, debug info will be printed\n"
          "  -l redirect stdout to a file\n"
          "  -L redirect stderr to a file\n");
  exit(EXIT_FAILURE);
}

// Parse the command line flags and return the result in an Options structure
// passed as argument.
static void ParseCommandLine(int argc, char *const *argv, struct Options *opt) {
  extern char *optarg;
  extern int optind, optopt;
  int c;

  while ((c = getopt(argc, argv, ":CDS:W:t:T:M:m:l:L:")) != -1) {
    switch (c) {
      case 'C':
        // Shortcut for the "does this system support sandboxing" check.
        exit(CheckNamespacesSupported());
        break;
      case 'S':
        if (opt->sandbox_root == NULL) {
          char *sandbox_root = strdup(optarg);

          // Make sure that the sandbox_root path has no trailing slash.
          if (sandbox_root[strlen(sandbox_root) - 1] == '/') {
            sandbox_root[strlen(sandbox_root) - 1] = 0;
          }

          opt->sandbox_root = sandbox_root;
        } else {
          Usage(argc, argv,
                "Multiple sandbox roots (-S) specified, expected one.");
        }
        break;
      case 'W':
        if (opt->working_dir == NULL) {
          opt->working_dir = optarg;
        } else {
          Usage(argc, argv,
                "Multiple working directories (-W) specified, expected at most "
                "one.");
        }
        break;
      case 't':
        if (sscanf(optarg, "%lf", &opt->kill_delay_secs) != 1 ||
            opt->kill_delay_secs < 0) {
          Usage(argc, argv, "Invalid kill delay (-t) value: %lf",
                opt->kill_delay_secs);
        }
        break;
      case 'T':
        if (sscanf(optarg, "%lf", &opt->timeout_secs) != 1 ||
            opt->timeout_secs < 0) {
          Usage(argc, argv, "Invalid timeout (-T) value: %lf",
                opt->timeout_secs);
        }
        break;
      case 'M':
        if (optarg[0] != '/') {
          Usage(argc, argv, "The -M option must be used with absolute paths only.");
        }
        // The last -M flag wasn't followed by an -m flag, so assume that the source should be mounted in the sandbox in the same path as outside.
        if (opt->mount_sources[opt->num_mounts] != NULL) {
          opt->mount_targets[opt->num_mounts] = opt->mount_sources[opt->num_mounts];
          opt->num_mounts++;
        }
        opt->mount_sources[opt->num_mounts] = optarg;
        break;
      case 'm':
        if (optarg[0] != '/') {
          Usage(argc, argv, "The -m option must be used with absolute paths only.");
        }
        if (opt->mount_sources[opt->num_mounts] == NULL) {
          Usage(argc, argv, "The -m option must be preceded by an -M option.");
        }
        opt->mount_targets[opt->num_mounts++] = optarg;
        break;
      case 'D':
        global_debug = true;
        break;
      case 'l':
        if (opt->stdout_path == NULL) {
          opt->stdout_path = optarg;
        } else {
          Usage(argc, argv,
                "Cannot redirect stdout to more than one destination.");
        }
        break;
      case 'L':
        if (opt->stderr_path == NULL) {
          opt->stderr_path = optarg;
        } else {
          Usage(argc, argv,
                "Cannot redirect stderr to more than one destination.");
        }
        break;
      case '?':
        Usage(argc, argv, "Unrecognized argument: -%c (%d)", optopt, optind);
        break;
      case ':':
        Usage(argc, argv, "Flag -%c requires an argument", optopt);
        break;
    }
  }

  if (opt->sandbox_root == NULL) {
    Usage(argc, argv, "Sandbox root (-S) must be specified");
  }

  // The last -M flag wasn't followed by an -m flag, assume that the source should be mounted in the sandbox in the same path as outside.
  if (opt->mount_sources[opt->num_mounts] != NULL &&
      opt->mount_targets[opt->num_mounts] == NULL) {
    opt->mount_targets[opt->num_mounts] = opt->mount_sources[opt->num_mounts];
    opt->num_mounts++;
  }

  opt->args = argv + optind;
  if (argc <= optind) {
    Usage(argc, argv, "No command specified.");
  }
}

static void CreateNamespaces() {
  // This weird workaround is necessary due to unshare seldomly failing with
  // EINVAL due to a race condition in the Linux kernel (see
  // https://lkml.org/lkml/2015/7/28/833). An alternative would be to use
  // clone/waitpid instead.
  int delay = 1;
  int tries = 0;
  const int max_tries = 100;
  while (tries++ < max_tries) {
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC) ==
        0) {
      PRINT_DEBUG("unshare succeeded after %d tries\n", tries);
      return;
    } else {
      if (errno != EINVAL) {
        perror("unshare");
        exit(EXIT_FAILURE);
      }
    }

    // Exponential back-off, but sleep at most 250ms.
    usleep(delay);
    if (delay < 250000) {
      delay *= 2;
    }
  }
  fprintf(stderr,
          "unshare failed with EINVAL even after %d tries, giving up.\n",
          tries);
  exit(EXIT_FAILURE);
}

static void CreateFile(const char *path) {
  int handle;
  CHECK_CALL(handle = open(path, O_CREAT | O_WRONLY | O_EXCL, 0666));
  CHECK_CALL(close(handle));
}

static void SetupDevices() {
  CHECK_CALL(mkdir("dev", 0755));
  const char *devs[] = {"/dev/null", "/dev/random", "/dev/urandom", "/dev/zero",
                        NULL};
  for (int i = 0; devs[i] != NULL; i++) {
    CreateFile(devs[i] + 1);
    CHECK_CALL(mount(devs[i], devs[i] + 1, NULL, MS_BIND, NULL));
  }

  CHECK_CALL(symlink("/proc/self/fd", "dev/fd"));
}

// Recursively creates the file or directory specified in "path" and its parent
// directories.
static int CreateTarget(const char *path, bool is_directory) {
  if (path == NULL) {
    errno = EINVAL;
    return -1;
  }

  struct stat sb;
  // If the path already exists...
  if (stat(path, &sb) == 0) {
    if (is_directory && S_ISDIR(sb.st_mode)) {
      // and it's a directory and supposed to be a directory, we're done here.
      return 0;
    } else if (!is_directory && S_ISREG(sb.st_mode)) {
      // and it's a regular file and supposed to be one, we're done here.
      return 0;
    } else {
      // otherwise something is really wrong.
      errno = is_directory ? ENOTDIR : EEXIST;
      return -1;
    }
  } else {
    // If stat failed because of any error other than "the path does not exist",
    // this is an error.
    if (errno != ENOENT) {
      return -1;
    }
  }

  // Create the parent directory.
  CHECK_CALL(CreateTarget(dirname(strdupa(path)), true));

  if (is_directory) {
    CHECK_CALL(mkdir(path, 0755));
  } else {
    CreateFile(path);
  }

  return 0;
}

static void SetupDirectories(struct Options *opt) {
  // Mount the sandbox and go there.
  CHECK_CALL(mount(opt->sandbox_root, opt->sandbox_root, NULL,
                   MS_BIND | MS_NOSUID, NULL));
  CHECK_CALL(chdir(opt->sandbox_root));

  // Setup /dev.
  SetupDevices();

  CHECK_CALL(mkdir("proc", 0755));
  CHECK_CALL(mount("/proc", "proc", NULL, MS_REC | MS_BIND, NULL));

  CHECK_CALL(mkdir("tmp", 0755));
  CHECK_CALL(mount("tmpfs", "tmp", "tmpfs", MS_NOSUID | MS_NODEV,
                   "size=25%,mode=1777"));

  // Make sure the home directory exists and is writable.
  const char *homedir;
  if ((homedir = getenv("HOME")) == NULL) {
    homedir = getpwuid(getuid())->pw_dir;
  }

  if (homedir[0] != '/') {
    DIE("Home directory of user nobody must be an absolute path, but is %s",
        homedir);
  }

  char *homedir_absolute =
      malloc(strlen(opt->sandbox_root) + strlen(homedir) + 1);
  strcpy(homedir_absolute, opt->sandbox_root);
  strcat(homedir_absolute, homedir);

  CreateTarget(homedir_absolute, true);
  CHECK_CALL(mount("tmpfs", homedir_absolute, "tmpfs", MS_NOSUID | MS_NODEV,
                   "size=25%,mode=1777"));

  // Mount directories passed in argv
  for (int i = 0; i < opt->num_mounts; i++) {
    struct stat sb;
    stat(opt->mount_sources[i], &sb);

    if (global_debug) {
      if (strcmp(opt->mount_sources[i], opt->mount_targets[i]) == 0) {
        // The file is mounted to the same path inside the sandbox, as outside (e.g. /home/user -> <sandbox>/home/user), so we'll just show a simplified version of the mount command.
        PRINT_DEBUG("mount: %s\n", opt->mount_sources[i]);
      } else {
        // The file is mounted to a custom location inside the sandbox.
        // Create a user-friendly string for the sandboxed path and show it.
        char *user_friendly_mount_target =
            malloc(strlen("<sandbox>") + strlen(opt->mount_targets[i]) + 1);
        strcpy(user_friendly_mount_target, "<sandbox>");
        strcat(user_friendly_mount_target, opt->mount_targets[i]);
        PRINT_DEBUG("mount: %s -> %s\n", opt->mount_sources[i],
                    user_friendly_mount_target);
        free(user_friendly_mount_target);
      }
    }

    char *full_sandbox_path = malloc(strlen(opt->sandbox_root) + strlen(opt->mount_targets[i]) + 1);
    strcpy(full_sandbox_path, opt->sandbox_root);
    strcat(full_sandbox_path, opt->mount_targets[i]);
    CHECK_CALL(CreateTarget(full_sandbox_path, S_ISDIR(sb.st_mode)));
    CHECK_CALL(mount(opt->mount_sources[i], full_sandbox_path, NULL,
                     MS_REC | MS_BIND | MS_RDONLY, NULL));
  }
}

// Write the file "filename" using a format string specified by "fmt". Returns
// -1 on failure.
static int WriteFile(const char *filename, const char *fmt, ...) {
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

static void SetupUserNamespace(int uid, int gid) {
  // Disable needs for CAP_SETGID
  int r = WriteFile("/proc/self/setgroups", "deny");
  if (r < 0 && errno != ENOENT) {
    // Writing to /proc/self/setgroups might fail on earlier
    // version of linux because setgroups does not exist, ignore.
    perror("WriteFile(\"/proc/self/setgroups\", \"deny\")");
    exit(EXIT_FAILURE);
  }

  // Set group and user mapping from outer namespace to inner:
  // No changes in the parent, be nobody in the child.
  //
  // We can't be root in the child, because some code may assume that running as
  // root grants it certain capabilities that it doesn't in fact have. It's
  // safer to let the child think that it is just a normal user.
  CHECK_CALL(WriteFile("/proc/self/uid_map", "%d %d 1\n", kNobodyUid, uid));
  CHECK_CALL(WriteFile("/proc/self/gid_map", "%d %d 1\n", kNobodyGid, gid));

  CHECK_CALL(setresuid(kNobodyUid, kNobodyUid, kNobodyUid));
  CHECK_CALL(setresgid(kNobodyGid, kNobodyGid, kNobodyGid));
}

static void ChangeRoot(struct Options *opt) {
  // move the real root to old_root, then detach it
  char old_root[16] = "old-root-XXXXXX";
  if (mkdtemp(old_root) == NULL) {
    perror("mkdtemp");
    DIE("mkdtemp returned NULL\n");
  }

  // pivot_root has no wrapper in libc, so we need syscall()
  CHECK_CALL(syscall(SYS_pivot_root, ".", old_root));
  CHECK_CALL(chroot("."));
  CHECK_CALL(umount2(old_root, MNT_DETACH));
  CHECK_CALL(rmdir(old_root));

  if (opt->working_dir != NULL) {
    CHECK_CALL(chdir(opt->working_dir));
  }
}

// Called when timeout or signal occurs.
void OnSignal(int sig) {
  global_signal = sig;

  // Nothing to do if we received a signal before spawning the child.
  if (global_child_pid == -1) {
    return;
  }

  if (sig == SIGALRM) {
    // SIGALRM represents a timeout, so we should give the process a bit of
    // time to die gracefully if it needs it.
    KillEverything(global_child_pid, true, global_kill_delay);
  } else {
    // Signals should kill the process quickly, as it's typically blocking
    // the return of the prompt after a user hits "Ctrl-C".
    KillEverything(global_child_pid, false, global_kill_delay);
  }
}

// Run the command specified by the argv array and kill it after timeout
// seconds.
static void SpawnCommand(char *const *argv, double timeout_secs) {
  for (int i = 0; argv[i] != NULL; i++) {
    PRINT_DEBUG("arg: %s\n", argv[i]);
  }

  CHECK_CALL(global_child_pid = fork());
  if (global_child_pid == 0) {
    // In child.
    CHECK_CALL(setsid());
    ClearSignalMask();

    // Force umask to include read and execute for everyone, to make
    // output permissions predictable.
    umask(022);

    // Does not return unless something went wrong.
    CHECK_CALL(execvp(argv[0], argv));
  } else {
    // In parent.

    // Set up a signal handler which kills all subprocesses when the given
    // signal is triggered.
    HandleSignal(SIGALRM, OnSignal);
    HandleSignal(SIGTERM, OnSignal);
    HandleSignal(SIGINT, OnSignal);
    SetTimeout(timeout_secs);

    int status = WaitChild(global_child_pid, argv[0]);

    // The child is done for, but may have grandchildren that we still have to
    // kill.
    kill(-global_child_pid, SIGKILL);

    if (global_signal > 0) {
      // Don't trust the exit code if we got a timeout or signal.
      UnHandle(global_signal);
      raise(global_signal);
    } else if (WIFEXITED(status)) {
      exit(WEXITSTATUS(status));
    } else {
      int sig = WTERMSIG(status);
      UnHandle(sig);
      raise(sig);
    }
  }
}

int main(int argc, char *const argv[]) {
  struct Options opt;
  memset(&opt, 0, sizeof(opt));
  opt.mount_sources = calloc(argc, sizeof(char *));
  opt.mount_targets = calloc(argc, sizeof(char *));

  ParseCommandLine(argc, argv, &opt);
  global_kill_delay = opt.kill_delay_secs;

  int uid = SwitchToEuid();
  int gid = SwitchToEgid();

  RedirectStdout(opt.stdout_path);
  RedirectStderr(opt.stderr_path);

  PRINT_DEBUG("sandbox root is %s\n", opt.sandbox_root);
  PRINT_DEBUG("working dir is %s\n",
              (opt.working_dir != NULL) ? opt.working_dir : "/ (default)");

  CreateNamespaces();

  // Make our mount namespace private, so that further mounts do not affect the
  // outside environment.
  CHECK_CALL(mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL));

  SetupDirectories(&opt);
  SetupUserNamespace(uid, gid);
  ChangeRoot(&opt);

  SpawnCommand(opt.args, opt.timeout_secs);

  free(opt.mount_sources);
  free(opt.mount_targets);

  return 0;
}
