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

#include <errno.h>  // errno, ENAMETOOLONG
#include <limits.h>
#include <linux/magic.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // strerror
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;
using std::string;
using std::vector;

string GetOutputRoot() {
  char buf[2048];
  string base;
  const char* home = getenv("HOME");
  if (home != NULL) {
    base = home;
  } else {
    struct passwd pwbuf;
    struct passwd *pw = NULL;
    int uid = getuid();
    int r = getpwuid_r(uid, &pwbuf, buf, 2048, &pw);
    if (r != -1 && pw != NULL) {
      base = pw->pw_dir;
    }
  }

  if (base != "") {
    return blaze_util::JoinPath(base, ".cache/bazel");
  }

  return "/tmp";
}

void WarnFilesystemType(const string& output_base) {
  struct statfs buf = {};
  if (statfs(output_base.c_str(), &buf) < 0) {
    fprintf(stderr,
            "WARNING: couldn't get file system type information for '%s': %s\n",
            output_base.c_str(), strerror(errno));
    return;
  }

  if (buf.f_type == NFS_SUPER_MAGIC) {
    fprintf(stderr, "WARNING: Output base '%s' is on NFS. This may lead "
            "to surprising failures and undetermined behavior.\n",
            output_base.c_str());
  }
}

string GetSelfPath() {
  char buffer[PATH_MAX] = {};
  ssize_t bytes = readlink("/proc/self/exe", buffer, sizeof(buffer));
  if (bytes == sizeof(buffer)) {
    // symlink contents truncated
    bytes = -1;
    errno = ENAMETOOLONG;
  }
  if (bytes == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error reading /proc/self/exe");
  }
  buffer[bytes] = '\0';  // readlink does not NUL-terminate
  return string(buffer);
}

pid_t GetPeerProcessId(int socket) {
  struct ucred creds = {};
  socklen_t len = sizeof creds;
  if (getsockopt(socket, SOL_SOCKET, SO_PEERCRED, &creds, &len) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "can't get server pid from connection");
  }
  return creds.pid;
}

uint64_t MonotonicClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

uint64_t ProcessClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  if (batch_cpu_scheduling) {
    sched_param param = {};
    param.sched_priority = 0;
    if (sched_setscheduler(0, SCHED_BATCH, &param)) {
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "sched_setscheduler(SCHED_BATCH) failed");
    }
  }

  if (io_nice_level >= 0) {
    if (blaze_util::sys_ioprio_set(
            IOPRIO_WHO_PROCESS, getpid(),
            IOPRIO_PRIO_VALUE(IOPRIO_CLASS_BE, io_nice_level)) < 0) {
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "ioprio_set() with class %d and level %d failed",
           IOPRIO_CLASS_BE, io_nice_level);
    }
  }
}

string GetProcessCWD(int pid) {
  char server_cwd[PATH_MAX] = {};
  if (readlink(
          ("/proc/" + ToString(pid) + "/cwd").c_str(),
          server_cwd, sizeof(server_cwd)) < 0) {
    return "";
  }

  return string(server_cwd);
}

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".so");
}

string GetDefaultHostJavabase() {
  // if JAVA_HOME is defined, then use it as default.
  const char *javahome = getenv("JAVA_HOME");
  if (javahome != NULL) {
    return string(javahome);
  }

  // which javac
  string javac_dir = blaze_util::Which("javac");
  if (javac_dir.empty()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "Could not find javac");
  }

  // Resolve all symlinks.
  char resolved_path[PATH_MAX];
  if (realpath(javac_dir.c_str(), resolved_path) == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Could not resolve javac directory");
  }
  javac_dir = resolved_path;

  // dirname dirname
  return blaze_util::Dirname(blaze_util::Dirname(javac_dir));
}

// Called from a signal handler. Therefore, we can't use our usual set of
// helper functions for reading files, splitting strings and so on.
static bool GetStartTime(int pid, char* output, int output_len) {
  char statfile[128];
  snprintf(statfile, sizeof(statfile), "/proc/%d/stat", pid);
  int fd = open(statfile, O_RDONLY);
  if (fd < 0) {
    return false;
  }

  // Note that this allocates 1K on any random stack the signal handler is
  // called on
  char statline[1024];
  int statline_len = read(fd, statline, 1024);
  close(fd);
  if (statline_len < 0) {
    return false;
  }

  // Field 22 is that start time of the process since system startup in jiffies.
  int space_count = 0;
  int space_21 = -1;
  int space_22 = -1;

  for (int i = 0; i < statline_len; i++) {
    if (statline[i] == ' ') {
      switch (++space_count) {
        case 21:
          space_21 = i;
          break;

        case 22:
          space_22 = i;
          break;

        default:
          // We don't care
          break;
      }
    }
  }

  if (space_21 == -1 || space_22 == -1) {
    // Invalid statline format
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "Format of stat file at %s is unknown", statfile);
  }

  int jiffies_len = space_22 - space_21 - 1;
  if (jiffies_len >= output_len) {
    // Not enough space in output buffer (Note that we need one extra byte
    // for the terminating NUL!)
    return false;
  }

  strncpy(output, statline + space_21 + 1, jiffies_len);
  output[jiffies_len] = 0;
  return true;
}

void WriteSystemSpecificProcessIdentifier(const string& server_dir) {
  char start_time[256];
  if (!GetStartTime(getpid(), start_time, 256)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "Cannot get start time of process %d", getpid());
  }

  string start_time_file = blaze_util::JoinPath(server_dir, "server.starttime");
  if (!WriteFile(start_time, start_time_file)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "Cannot write start time in server dir %s", server_dir.c_str());
  }
}

// On Linux we use a combination of PID and start time to identify the server
// process. That is supposed to be unique unless one can start more processes
// than there are PIDs available within a single jiffy.
//
// This looks complicated, but all it does is an open(), then read(), then
// close(), all of which are safe to call from signal handlers.
bool KillServerProcess(
    int pid, const string& output_base, const string& install_base) {
  char start_time[256];
  if (!GetStartTime(pid, start_time, sizeof(start_time))) {
    // Cannot read PID file from /proc . Process died meantime, all is good. No
    // stale server is present.
    return false;
  }

  string recorded_start_time;
  bool file_present = ReadFile(
      blaze_util::JoinPath(output_base, "server/server.starttime"),
      &recorded_start_time);

  // start time file got deleted, but PID file didn't. This is strange.
  // Assume that this is an old Blaze process that doesn't know how to write
  // start time files yet.
  if (file_present && recorded_start_time != start_time) {
    // This is a different process.
    return false;
  }

  // Kill the process and make sure it's dead before proceeding.
  killpg(pid, SIGKILL);
  int check_killed_retries = 10;
  while (killpg(pid, 0) == 0) {
    if (check_killed_retries-- > 0) {
      sleep(1);
    } else {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "Attempted to kill stale blaze server process (pid=%d) using "
          "SIGKILL, but it did not die in a timely fashion.", pid);
    }
  }
  return true;
}

// Not supported.
void ExcludePathFromBackup(const string &path) {
}

}  // namespace blaze
