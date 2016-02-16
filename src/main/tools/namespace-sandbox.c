// Copyright 2014 The Bazel Authors. All rights reserved.
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
#include <limits.h>
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

#include "network-tools.h"
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

// The username of 'nobody'.
static const char *kNobodyUsername = "nobody";

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
  char **mount_sources;      // Map of directories to mount, from (-M)
  char **mount_targets;      // sources -> targets (-m)
  size_t mount_map_sizes;    // How many elements in mount_{sources,targets}
  int num_mounts;            // How many mounts were specified
  char **create_dirs;        // empty dirs to create (-d)
  size_t create_dirs_size;   // How many elements in create_dirs
  int num_create_dirs;       // How many empty dirs to create were specified
  int fake_root;             // Pretend to be root inside the namespace.
  int create_netns;          // If 1, create a new network namespace.
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
                             CLONE_NEWIPC | CLONE_NEWNET | SIGCHLD,
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

  fprintf(stderr, "\nUsage: %s [-S sandbox-root] -- command arg1\n", argv[0]);
  fprintf(stderr, "  provided:");
  for (i = 0; i < argc; i++) {
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(
      stderr,
      "\nMandatory arguments:\n"
      "  -S <sandbox-root>  directory which will become the root of the "
      "sandbox\n"
      "  --  command to run inside sandbox, followed by arguments\n"
      "\n"
      "Optional arguments:\n"
      "  -W <working-dir>  working directory\n"
      "  -T <timeout>  timeout after which the child process will be "
      "terminated with SIGTERM\n"
      "  -t <timeout>  in case timeout occurs, how long to wait before killing "
      "the child with SIGKILL\n"
      "  -d <dir>  create an empty directory in the sandbox\n"
      "  -M/-m <source/target>  system directory to mount inside the sandbox\n"
      "    Multiple directories can be specified and each of them will be "
      "mounted readonly.\n"
      "    The -M option specifies which directory to mount, the -m option "
      "specifies where to\n"
      "    mount it in the sandbox.\n"
      "  -n if set, a new network namespace will be created\n"
      "  -r if set, make the uid/gid be root, otherwise use nobody\n"
      "  -D  if set, debug info will be printed\n"
      "  -l <file>  redirect stdout to a file\n"
      "  -L <file>  redirect stderr to a file\n"
      "  @FILE read newline-separated arguments from FILE\n");
  exit(EXIT_FAILURE);
}

// Deals with an unfinished (source but no target) mapping in opt.
// Also adds a new unfinished mapping if source is not NULL.
static void AddMountSource(char *source, struct Options *opt) {
  // The last -M flag wasn't followed by an -m flag, so assume that the source
  // should be mounted in the sandbox in the same path as outside.
  if (opt->mount_sources[opt->num_mounts] != NULL) {
    opt->mount_targets[opt->num_mounts] = opt->mount_sources[opt->num_mounts];
    opt->num_mounts++;
  }
  if (source != NULL) {
    if (opt->num_mounts >= opt->mount_map_sizes - 1) {
      opt->mount_sources = realloc(opt->mount_sources,
                                   opt->mount_map_sizes * sizeof(char *) * 2);
      if (opt->mount_sources == NULL) {
        DIE("realloc failed\n");
      }
      memset(opt->mount_sources + opt->mount_map_sizes, 0,
             opt->mount_map_sizes * sizeof(char *));
      opt->mount_targets = realloc(opt->mount_targets,
                                   opt->mount_map_sizes * sizeof(char *) * 2);
      if (opt->mount_targets == NULL) {
        DIE("realloc failed\n");
      }
      memset(opt->mount_targets + opt->mount_map_sizes, 0,
             opt->mount_map_sizes * sizeof(char *));
      opt->mount_map_sizes *= 2;
    }
    opt->mount_sources[opt->num_mounts] = source;
  }
}

static void AddCreateDir(char *create_dir, struct Options *opt) {
  if (opt->num_create_dirs > opt->create_dirs_size - 1) {
    opt->create_dirs =
        realloc(opt->create_dirs, opt->create_dirs_size * sizeof(char *) * 2);
    if (opt->create_dirs == NULL) {
      DIE("realloc failed\n");
    }
    memset(opt->create_dirs + opt->create_dirs_size, 0,
           opt->create_dirs_size * sizeof(char *));
    opt->create_dirs_size *= 2;
  }
  opt->create_dirs[opt->num_create_dirs++] = create_dir;
}

static void ParseCommandLine(int argc, char *const *argv, struct Options *opt);

// Parses command line flags from a file named filename.
// Expects optind to be initialized to 0 before being called.
static void ParseOptionsFile(const char *filename, struct Options *opt) {
  FILE *const options_file = fopen(filename, "rb");
  if (options_file == NULL) {
    DIE("opening argument file %s failed\n", filename);
  }
  size_t sub_argv_size = 20;
  char **sub_argv = malloc(sizeof(char *) * sub_argv_size);
  sub_argv[0] = "";
  int sub_argc = 1;

  bool done = false;
  while (!done) {
    // This buffer determines the maximum size of arguments we can handle out of
    // the file. We DIE down below if it's ever too short.
    // 4096 is a common value for PATH_MAX. However, many filesystems support
    // arbitrarily long pathnames, so this might not be long enough to handle an
    // arbitrary filename no matter what. Twice the usual PATH_MAX seems
    // reasonable for now.
    char argument[8192];
    if (fgets(argument, sizeof(argument), options_file) == NULL) {
      if (feof(options_file)) {
        done = true;
        continue;
      } else {
        DIE("reading from argument file %s failed\n", filename);
      }
    }
    const size_t length = strlen(argument);
    if (length == 0) continue;
    if (length == sizeof(argument)) {
      DIE("argument from file %s is too long (> %zu)\n", filename,
          sizeof(argument));
    }
    if (argument[length - 1] == '\n') {
      argument[length - 1] = '\0';
    } else {
      done = true;
    }
    if (sub_argv_size == sub_argc + 1) {
      sub_argv_size *= 2;
      sub_argv = realloc(sub_argv, sizeof(char *) * sub_argv_size);
    }
    sub_argv[sub_argc++] = strdup(argument);
  }
  if (fclose(options_file) != 0) {
    DIE("closing options file %s failed\n", filename);
  }
  sub_argv[sub_argc] = NULL;

  ParseCommandLine(sub_argc, sub_argv, opt);
}

// Parse the command line flags and return the result in an Options structure
// passed as argument.
static void ParseCommandLine(int argc, char *const *argv, struct Options *opt) {
  extern char *optarg;
  extern int optind, optopt;
  int c;

  while ((c = getopt(argc, argv, ":CDd:l:L:m:M:nrt:T:S:W:")) != -1) {
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
      case 'd':
        if (optarg[0] != '/') {
          Usage(argc, argv,
                "The -d option must be used with absolute paths only.");
        }
        AddCreateDir(optarg, opt);
        break;
      case 'M':
        if (optarg[0] != '/') {
          Usage(argc, argv,
                "The -M option must be used with absolute paths only.");
        }
        AddMountSource(optarg, opt);
        break;
      case 'm':
        if (optarg[0] != '/') {
          Usage(argc, argv,
                "The -m option must be used with absolute paths only.");
        }
        if (opt->mount_sources[opt->num_mounts] == NULL) {
          Usage(argc, argv, "The -m option must be preceded by an -M option.");
        }
        opt->mount_targets[opt->num_mounts++] = optarg;
        break;
      case 'n':
        opt->create_netns = 1;
        break;
      case 'r':
        opt->fake_root = 1;
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

  AddMountSource(NULL, opt);

  while (optind < argc && argv[optind][0] == '@') {
    const char *filename = argv[optind] + 1;
    const int old_optind = optind;
    optind = 0;
    ParseOptionsFile(filename, opt);
    optind = old_optind + 1;
  }

  if (argc > optind) {
    if (opt->args == NULL) {
      opt->args = argv + optind;
    } else {
      Usage(argc, argv, "Merging commands not supported.");
    }
  }
}

static void CreateNamespaces(int create_netns) {
  // This weird workaround is necessary due to unshare seldomly failing with
  // EINVAL due to a race condition in the Linux kernel (see
  // https://lkml.org/lkml/2015/7/28/833). An alternative would be to use
  // clone/waitpid instead.
  int delay = 1;
  int tries = 0;
  const int max_tries = 100;
  while (tries++ < max_tries) {
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC |
                (create_netns ? CLONE_NEWNET : 0)) == 0) {
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

// Creates an empty file at 'path' by hard linking it from a known empty file.
// This is over two times faster than creating empty files via open() on
// certain filesystems (e.g. XFS).
static void LinkFile(const char *path) {
  CHECK_CALL(link("tmp/empty_file", path));
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
    LinkFile(path);
  }

  return 0;
}

static void SetupDevices() {
  CHECK_CALL(CreateTarget("dev", true));
  const char *devs[] = {"/dev/null", "/dev/random", "/dev/urandom", "/dev/zero",
                        NULL};
  for (int i = 0; devs[i] != NULL; i++) {
    LinkFile(devs[i] + 1);
    CHECK_CALL(mount(devs[i], devs[i] + 1, NULL, MS_BIND, NULL));
  }

  CHECK_CALL(symlink("/proc/self/fd", "dev/fd"));
}

static void SetupDirectories(struct Options *opt) {
  // Mount the sandbox and go there.
  CHECK_CALL(mount(opt->sandbox_root, opt->sandbox_root, NULL,
                   MS_BIND | MS_NOSUID, NULL));
  CHECK_CALL(chdir(opt->sandbox_root));

  // This is used as the base for hardlinking the input files.
  CHECK_CALL(CreateTarget("tmp", true));
  CreateFile("tmp/empty_file");

  // Setup /dev.
  SetupDevices();

  CHECK_CALL(CreateTarget("proc", true));
  CHECK_CALL(mount("/proc", "proc", NULL, MS_REC | MS_BIND, NULL));

  // Make sure the home directory exists, too.
  char *homedir_from_env = getenv("HOME");
  if (homedir_from_env != NULL) {
    if (homedir_from_env[0] != '/') {
      DIE("Home directory specified in $HOME must be an absolute path, but is "
          "%s",
          homedir_from_env);
    }
    if (strcmp(homedir_from_env, "/") != 0) {
      AddCreateDir(homedir_from_env, opt);
    }
  }

  char *homedir = getpwuid(getuid())->pw_dir;
  if (homedir != NULL &&
      (homedir_from_env == NULL || strcmp(homedir_from_env, homedir) != 0)) {
    if (homedir[0] != '/') {
      DIE("Home directory of user nobody must be an absolute path, but is %s",
          homedir);
    }
    if (strcmp(homedir, "/") != 0) {
      AddCreateDir(homedir, opt);
    }
  }

  // Create needed directories.
  for (int i = 0; i < opt->num_create_dirs; i++) {
    if (global_debug) {
      PRINT_DEBUG("createdir: %s\n", opt->create_dirs[i]);
    }
    CHECK_CALL(CreateTarget(opt->create_dirs[i] + 1, true));
  }

  // Mount all mounts.
  for (int i = 0; i < opt->num_mounts; i++) {
    struct stat sb;
    stat(opt->mount_sources[i], &sb);

    if (global_debug) {
      if (strcmp(opt->mount_sources[i], opt->mount_targets[i]) == 0) {
        // The file is mounted to the same path inside the sandbox, as outside
        // (e.g. /home/user -> <sandbox>/home/user), so we'll just show a
        // simplified version of the mount command.
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

    char *full_sandbox_path =
        malloc(strlen(opt->sandbox_root) + strlen(opt->mount_targets[i]) + 1);
    strcpy(full_sandbox_path, opt->sandbox_root);
    strcat(full_sandbox_path, opt->mount_targets[i]);
    CHECK_CALL(CreateTarget(full_sandbox_path, S_ISDIR(sb.st_mode)));
    CHECK_CALL(mount(opt->mount_sources[i], full_sandbox_path, NULL,
                     MS_REC | MS_BIND | MS_RDONLY, NULL));
    free(full_sandbox_path);
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

static void SetupUserNamespace(int uid, int gid, int new_uid, int new_gid) {
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
  CHECK_CALL(WriteFile("/proc/self/uid_map", "%d %d 1\n", new_uid, uid));
  CHECK_CALL(WriteFile("/proc/self/gid_map", "%d %d 1\n", new_gid, gid));

  CHECK_CALL(setresuid(new_uid, new_uid, new_uid));
  CHECK_CALL(setresgid(new_gid, new_gid, new_gid));
}

static void SetupUserNamespaceForNobody(int uid, int gid) {
  struct passwd *pwd = getpwnam(kNobodyUsername);

  if (pwd == NULL) {
    perror("Unable to find passwd entry for user nobody.");
    exit(EXIT_FAILURE);
  }

  SetupUserNamespace(uid, gid, pwd->pw_uid, pwd->pw_gid);
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
  // 16 elements is a sane default, will be realloc'd as needed anyway.
  opt.mount_sources = calloc(16, sizeof(char *));
  opt.mount_targets = calloc(16, sizeof(char *));
  opt.mount_map_sizes = 16;
  // We'll need at least two slots for homedir_from_env and homedir.
  opt.create_dirs = calloc(2, sizeof(char *));
  opt.create_dirs_size = 2;

  ParseCommandLine(argc, argv, &opt);
  if (opt.args == NULL) {
    Usage(argc, argv, "No command specified.");
  }
  if (opt.sandbox_root == NULL) {
    Usage(argc, argv, "Sandbox root (-S) must be specified");
  }
  global_kill_delay = opt.kill_delay_secs;

  int uid = SwitchToEuid();
  int gid = SwitchToEgid();

  RedirectStdout(opt.stdout_path);
  RedirectStderr(opt.stderr_path);

  PRINT_DEBUG("sandbox root is %s\n", opt.sandbox_root);
  PRINT_DEBUG("working dir is %s\n",
              (opt.working_dir != NULL) ? opt.working_dir : "/ (default)");

  CreateNamespaces(opt.create_netns);
  if (opt.create_netns) {
    // Enable the loopback interface because some application may want
    // to use it.
    BringupInterface("lo");
  }

  // Make our mount namespace private, so that further mounts do not affect the
  // outside environment.
  CHECK_CALL(mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL));

  SetupDirectories(&opt);
  if (opt.fake_root) {
    SetupUserNamespace(uid, gid, 0, 0);
  } else {
    SetupUserNamespaceForNobody(uid, gid);
  }
  ChangeRoot(&opt);

  SpawnCommand(opt.args, opt.timeout_secs);

  free(opt.create_dirs);
  free(opt.mount_sources);
  free(opt.mount_targets);

  return 0;
}
