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
 * This is PID 1 inside the sandbox environment and runs in a separate user,
 * mount, UTS, IPC and PID namespace.
 */

#include "src/main/tools/linux-sandbox-pid1.h"

#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <math.h>
#include <mntent.h>
#include <net/if.h>
#include <pwd.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#ifndef MS_REC
// Some systems do not define MS_REC in sys/mount.h. We might be able to grab it
// from linux/fs.h instead (cf. #2667).
#include <linux/fs.h>
#endif

#include "src/main/tools/linux-sandbox-options.h"
#include "src/main/tools/linux-sandbox.h"
#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"

static int global_child_pid;

static void SetupSelfDestruction(int *sync_pipe) {
  // We could also poll() on the pipe fd to find out when the parent goes away,
  // and rely on SIGCHLD interrupting that otherwise. That might require us to
  // install some trivial handler for SIGCHLD. Using O_ASYNC to turn the pipe
  // close into SIGIO may also work. Another option is signalfd, although that's
  // almost as obscure as this prctl.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0) {
    DIE("prctl");
  }

  // Switch to a new process group, otherwise our process group will still refer
  // to the outer PID namespace. We might then accidentally kill our parent by a
  // call to e.g. `kill(0, sig)`.
  if (setpgid(0, 0) < 0) {
    DIE("setpgid");
  }

  // Verify that the parent still lives.
  char buf = 0;
  if (close(sync_pipe[0]) < 0) {
    DIE("close");
  }
  if (write(sync_pipe[1], &buf, 1) < 0) {
    DIE("write");
  }
  if (close(sync_pipe[1]) < 0) {
    DIE("close");
  }
}

static void SetupMountNamespace() {
  // Fully isolate our mount namespace private from outside events, so that
  // mounts in the outside environment do not affect our sandbox.
  if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) < 0) {
    DIE("mount");
  }
}

static void WriteFile(const std::string &filename, const char *fmt, ...) {
  FILE *stream = fopen(filename.c_str(), "w");
  if (stream == nullptr) {
    DIE("fopen(%s)", filename.c_str());
  }

  va_list ap;
  va_start(ap, fmt);
  int r = vfprintf(stream, fmt, ap);
  va_end(ap);

  if (r < 0) {
    DIE("vfprintf");
  }

  if (fclose(stream) != 0) {
    DIE("fclose(%s)", filename.c_str());
  }
}

static void SetupUserNamespace() {
  // Disable needs for CAP_SETGID.
  struct stat sb;
  if (stat("/proc/self/setgroups", &sb) == 0) {
    WriteFile("/proc/self/setgroups", "deny");
  } else {
    // Ignore ENOENT, because older Linux versions do not have this file (but
    // also do not require writing to it).
    if (errno != ENOENT) {
      DIE("stat(/proc/self/setgroups");
    }
  }

  int inner_uid, inner_gid;
  if (opt.fake_root) {
    // Change our username to 'root'.
    inner_uid = 0;
    inner_gid = 0;
  } else if (opt.fake_username) {
    // Change our username to 'nobody'.
    struct passwd *pwd = getpwnam("nobody");
    if (pwd == nullptr) {
      DIE("unable to find passwd entry for user nobody")
    }

    inner_uid = pwd->pw_uid;
    inner_gid = pwd->pw_gid;
  } else {
    // Do not change the username inside the sandbox.
    inner_uid = global_outer_uid;
    inner_gid = global_outer_gid;
  }

  WriteFile("/proc/self/uid_map", "%d %d 1\n", inner_uid, global_outer_uid);
  WriteFile("/proc/self/gid_map", "%d %d 1\n", inner_gid, global_outer_gid);
}

static void SetupUtsNamespace() {
  if (sethostname("localhost", 9) < 0) {
    DIE("sethostname");
  }

  if (setdomainname("localdomain", 11) < 0) {
    DIE("setdomainname");
  }
}

static void MountFilesystems() {
  for (const std::string &tmpfs_dir : opt.tmpfs_dirs) {
    PRINT_DEBUG("tmpfs: %s", tmpfs_dir.c_str());
    if (mount("tmpfs", tmpfs_dir.c_str(), "tmpfs",
              MS_NOSUID | MS_NODEV | MS_NOATIME, nullptr) < 0) {
      DIE("mount(tmpfs, %s, tmpfs, MS_NOSUID | MS_NODEV | MS_NOATIME, nullptr)",
          tmpfs_dir.c_str());
    }
  }

  // Make sure that our working directory is a mount point. The easiest way to
  // do this is by bind-mounting it upon itself.
  PRINT_DEBUG("working dir: %s", opt.working_dir.c_str());

  // An attempt to mount the sandbox in tmpfs will always fail, so this block is
  // slightly redundant with the next mount() check, but dumping the mount()
  // syscall is incredibly cryptic, so we explicitly check against and warn
  // about attempts to use tmpfs.
  for (const std::string &tmpfs_dir : opt.tmpfs_dirs) {
    if (opt.working_dir.find(tmpfs_dir) == 0) {
      DIE("The sandbox working directory cannot be below a path where we mount "
          "tmpfs (you requested mounting %s in %s). Is your --output_base= "
          "below one of your --sandbox_tmpfs_path values?",
          opt.working_dir.c_str(), tmpfs_dir.c_str());
    }
  }

  if (mount(opt.working_dir.c_str(), opt.working_dir.c_str(), nullptr, MS_BIND,
            nullptr) < 0) {
    DIE("mount(%s, %s, nullptr, MS_BIND, nullptr)", opt.working_dir.c_str(),
        opt.working_dir.c_str());
  }

  for (size_t i = 0; i < opt.bind_mount_sources.size(); i++) {
    const std::string &source = opt.bind_mount_sources.at(i);
    const std::string &target = opt.bind_mount_targets.at(i);
    PRINT_DEBUG("bind mount: %s -> %s", source.c_str(), target.c_str());
    if (mount(source.c_str(), target.c_str(), nullptr, MS_BIND, nullptr) < 0) {
      DIE("mount(%s, %s, nullptr, MS_BIND, nullptr)", source.c_str(),
          target.c_str());
    }
  }

  for (const std::string &writable_file : opt.writable_files) {
    PRINT_DEBUG("writable: %s", writable_file.c_str());
    if (mount(writable_file.c_str(), writable_file.c_str(), nullptr,
              MS_BIND | MS_REC, nullptr) < 0) {
      DIE("mount(%s, %s, nullptr, MS_BIND | MS_REC, nullptr)",
          writable_file.c_str(), writable_file.c_str());
    }
  }
}

// We later remount everything read-only, except the paths for which this method
// returns true.
static bool ShouldBeWritable(const std::string &mnt_dir) {
  if (mnt_dir == opt.working_dir) {
    return true;
  }

  for (const std::string &writable_file : opt.writable_files) {
    if (mnt_dir == writable_file) {
      return true;
    }
  }

  for (const std::string &tmpfs_dir : opt.tmpfs_dirs) {
    if (mnt_dir == tmpfs_dir) {
      return true;
    }
  }

  return false;
}

// Makes the whole filesystem read-only, except for the paths for which
// ShouldBeWritable returns true.
static void MakeFilesystemMostlyReadOnly() {
  FILE *mounts = setmntent("/proc/self/mounts", "r");
  if (mounts == nullptr) {
    DIE("setmntent");
  }

  struct mntent *ent;
  while ((ent = getmntent(mounts)) != nullptr) {
    int mountFlags = MS_BIND | MS_REMOUNT;

    // MS_REMOUNT does not allow us to change certain flags. This means, we have
    // to first read them out and then pass them in back again. There seems to
    // be no better way than this (an API for just getting the mount flags of a
    // mount entry as a bitmask would be great).
    if (hasmntopt(ent, "nodev") != nullptr) {
      mountFlags |= MS_NODEV;
    }
    if (hasmntopt(ent, "noexec") != nullptr) {
      mountFlags |= MS_NOEXEC;
    }
    if (hasmntopt(ent, "nosuid") != nullptr) {
      mountFlags |= MS_NOSUID;
    }
    if (hasmntopt(ent, "noatime") != nullptr) {
      mountFlags |= MS_NOATIME;
    }
    if (hasmntopt(ent, "nodiratime") != nullptr) {
      mountFlags |= MS_NODIRATIME;
    }
    if (hasmntopt(ent, "relatime") != nullptr) {
      mountFlags |= MS_RELATIME;
    }

    if (!ShouldBeWritable(ent->mnt_dir)) {
      mountFlags |= MS_RDONLY;
    }

    PRINT_DEBUG("remount %s: %s", (mountFlags & MS_RDONLY) ? "ro" : "rw",
                ent->mnt_dir);
    if (mount(nullptr, ent->mnt_dir, nullptr, mountFlags, nullptr) < 0) {
      // If we get EACCES or EPERM, this might be a mount-point for which we
      // don't have read access. Not much we can do about this, but it also
      // won't do any harm, so let's go on. The same goes for EINVAL or ENOENT,
      // which are fired in case a later mount overlaps an earlier mount, e.g.
      // consider the case of /proc, /proc/sys/fs/binfmt_misc and /proc, with
      // the latter /proc being the one that an outer sandbox has mounted on
      // top of its parent /proc. In that case, we're not allowed to remount
      // /proc/sys/fs/binfmt_misc, because it is hidden. If we get ESTALE, the
      // mount is a broken NFS mount. In the ideal case, the user would either
      // fix or remove that mount, but in cases where that's not possible, we
      // should just ignore it.
      switch (errno) {
        case EACCES:
        case EPERM:
        case EINVAL:
        case ENOENT:
        case ESTALE:
          PRINT_DEBUG(
              "remount(nullptr, %s, nullptr, %d, nullptr) failure (%m) ignored",
              ent->mnt_dir, mountFlags);
          break;
        default:
          DIE("remount(nullptr, %s, nullptr, %d, nullptr)", ent->mnt_dir,
              mountFlags);
      }
    }
  }

  endmntent(mounts);
}

static void MountProc() {
  // Mount a new proc on top of the old one, because the old one still refers to
  // our parent PID namespace.
  if (mount("/proc", "/proc", "proc", MS_NODEV | MS_NOEXEC | MS_NOSUID,
            nullptr) < 0) {
    DIE("mount");
  }
}

static void SetupNetworking() {
  // When running in a separate network namespace, enable the loopback interface
  // because some application may want to use it.
  if (opt.create_netns) {
    int fd;
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
      DIE("socket");
    }

    struct ifreq ifr = {};
    strncpy(ifr.ifr_name, "lo", IF_NAMESIZE);

    // Verify that name is valid.
    if (if_nametoindex(ifr.ifr_name) == 0) {
      DIE("if_nametoindex");
    }

    // Enable the interface.
    ifr.ifr_flags |= IFF_UP;
    if (ioctl(fd, SIOCSIFFLAGS, &ifr) < 0) {
      DIE("ioctl");
    }

    if (close(fd) < 0) {
      DIE("close");
    }
  }
}

static void EnterSandbox() {
  if (chdir(opt.working_dir.c_str()) < 0) {
    DIE("chdir(%s)", opt.working_dir.c_str());
  }
}

static void ForwardSignal(int signum) {
  PRINT_DEBUG("ForwardSignal(%d)", signum);
  kill(-global_child_pid, signum);
}

static void SpawnChild() {
  PRINT_DEBUG("calling fork...");
  global_child_pid = fork();

  if (global_child_pid < 0) {
    DIE("fork()");
  } else if (global_child_pid == 0) {
    // Put the child into its own process group.
    if (setpgid(0, 0) < 0) {
      DIE("setpgid");
    }

    // Try to assign our terminal to the child process.
    if (tcsetpgrp(STDIN_FILENO, getpgrp()) < 0 && errno != ENOTTY) {
      DIE("tcsetpgrp")
    }

    // Unblock all signals, restore default handlers.
    ClearSignalMask();

    // Force umask to include read and execute for everyone, to make output
    // permissions predictable.
    umask(022);

    // argv[] passed to execve() must be a null-terminated array.
    opt.args.push_back(nullptr);

    if (execvp(opt.args[0], opt.args.data()) < 0) {
      DIE("execvp(%s, %p)", opt.args[0], opt.args.data());
    }
  } else {
    PRINT_DEBUG("child started with PID %d", global_child_pid);
  }
}

static int WaitForChild() {
  while (true) {
    // Wait for some process to exit. This includes reparented processes in our
    // PID namespace.
    int status;
    const pid_t pid = TEMP_FAILURE_RETRY(wait(&status));

    if (pid < 0) {
      // We don't expect any errors besides EINTR. In particular, ECHILD should
      // be impossible because we haven't yet seen global_child_pid exit.
      DIE("wait");
    }

    PRINT_DEBUG("wait returned pid=%d, status=0x%02x", pid, status);

    // If this isn't our child's PID, there's nothing further to do; we've
    // successfully reaped a zombie.
    if (pid != global_child_pid) {
      continue;
    }

    // If the child exited due to a signal, log that fact and exit with the same
    // status.
    if (WIFSIGNALED(status)) {
      const int signal = WTERMSIG(status);
      PRINT_DEBUG("child exited due to signal %d", WTERMSIG(status));
      return 128 + signal;
    }

    // Otherwise it must have exited normally.
    const int exit_code = WEXITSTATUS(status);
    PRINT_DEBUG("child exited normally with code %d", exit_code);
    return exit_code;
  }
}

int Pid1Main(void *sync_pipe_param) {
  PRINT_DEBUG("Pid1Main started");

  if (getpid() != 1) {
    DIE("Using PID namespaces, but we are not PID 1");
  }

  // Start with default signal handlers and an empty signal mask.
  ClearSignalMask();

  SetupSelfDestruction(reinterpret_cast<int *>(sync_pipe_param));

  // Sandbox ourselves.
  SetupMountNamespace();
  SetupUserNamespace();
  if (opt.fake_hostname) {
    SetupUtsNamespace();
  }
  MountFilesystems();
  MakeFilesystemMostlyReadOnly();
  MountProc();
  SetupNetworking();
  EnterSandbox();

  // Ignore terminal signals; we hand off the terminal to the child in
  // SpawnChild below.
  IgnoreSignal(SIGTTIN);
  IgnoreSignal(SIGTTOU);

  // Fork the child process.
  SpawnChild();

  // Forward requests to shut down gracefully to the child.
  InstallSignalHandler(SIGTERM, ForwardSignal);

  // Note that there's no need to kill any remaining descendant processes; they
  // are in our PID namespace and the kernel will send them SIGKILL
  // automatically once we exit.
  return WaitForChild();
}
