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
#include <features.h>
#if !defined(__GLIBC__) || __GLIBC_PREREQ(2, 9)
#include <linux/magic.h>
#else
#define NFS_SUPER_MAGIC 0x6969
#endif
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // strerror
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using blaze_util::GetLastErrorString;
using std::string;
using std::vector;

string GetOutputRoot() {
  string base;
  string home = GetHomeDir();
  if (!home.empty()) {
    base = home;
  } else {
    char buf[2048];
    struct passwd pwbuf;
    struct passwd *pw = NULL;
    int uid = getuid();
    int r = getpwuid_r(uid, &pwbuf, buf, 2048, &pw);
    if (r != -1 && pw != NULL) {
      base = pw->pw_dir;
    }
  }

  if (!base.empty()) {
    return blaze_util::JoinPath(base, ".cache/bazel");
  }

  return "/tmp";
}

void WarnFilesystemType(const string& output_base) {
  struct statfs buf = {};
  if (statfs(output_base.c_str(), &buf) < 0) {
    BAZEL_LOG(WARNING) << "couldn't get file system type information for '"
                       << output_base << "': " << strerror(errno);
    return;
  }

  if (buf.f_type == NFS_SUPER_MAGIC) {
    BAZEL_LOG(WARNING) << "Output base '" << output_base
                       << "' is on NFS. This may lead to surprising failures "
                          "and undetermined behavior.";
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
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "error reading /proc/self/exe: " << GetLastErrorString();
  }
  buffer[bytes] = '\0';  // readlink does not NUL-terminate
  return string(buffer);
}

uint64_t GetMillisecondsMonotonic() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000LL + (ts.tv_nsec / 1000000LL);
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  if (batch_cpu_scheduling) {
    sched_param param = {};
    param.sched_priority = 0;
    if (sched_setscheduler(0, SCHED_BATCH, &param)) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "sched_setscheduler(SCHED_BATCH) failed: " << GetLastErrorString();
    }
  }

  if (io_nice_level >= 0) {
    if (blaze_util::sys_ioprio_set(
            IOPRIO_WHO_PROCESS, getpid(),
            IOPRIO_PRIO_VALUE(IOPRIO_CLASS_BE, io_nice_level)) < 0) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "ioprio_set() with class " << IOPRIO_CLASS_BE << " and level "
          << io_nice_level << " failed: " << GetLastErrorString();
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

static string Which(const string &executable) {
  string path(GetEnv("PATH"));
  if (path.empty()) {
    return "";
  }

  vector<string> pieces = blaze_util::Split(path, ':');
  for (auto piece : pieces) {
    if (piece.empty()) {
      piece = ".";
    }

    struct stat file_stat;
    string candidate = blaze_util::JoinPath(piece, executable);
    if (access(candidate.c_str(), X_OK) == 0 &&
        stat(candidate.c_str(), &file_stat) == 0 &&
        S_ISREG(file_stat.st_mode)) {
      return candidate;
    }
  }
  return "";
}

string GetSystemJavabase() {
  // if JAVA_HOME is defined, then use it as default.
  string javahome = GetEnv("JAVA_HOME");
  if (!javahome.empty()) {
    return javahome;
  }

  // which javac
  string javac_dir = Which("javac");
  if (javac_dir.empty()) {
    return "";
  }

  // Resolve all symlinks.
  char resolved_path[PATH_MAX];
  if (realpath(javac_dir.c_str(), resolved_path) == NULL) {
    return "";
  }
  javac_dir = resolved_path;

  // dirname dirname
  return blaze_util::Dirname(blaze_util::Dirname(javac_dir));
}

// Called from a signal handler!
static bool GetStartTime(const string& pid, string* start_time) {
  string statfile = "/proc/" + pid + "/stat";
  string statline;

  if (!blaze_util::ReadFile(statfile, &statline)) {
    return false;
  }

  vector<string> stat_entries = blaze_util::Split(statline, ' ');
  if (stat_entries.size() < 22) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Format of stat file at " << statfile
        << " is unknown: " << GetLastErrorString();
  }

  // Start time since startup in jiffies. This combined with the PID should be
  // unique.
  *start_time = stat_entries[21];
  return true;
}

void WriteSystemSpecificProcessIdentifier(
    const string& server_dir, pid_t server_pid) {
  string pid_string = ToString(server_pid);

  string start_time;
  if (!GetStartTime(pid_string, &start_time)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Cannot get start time of process " << pid_string << ": "
        << GetLastErrorString();
  }

  string start_time_file = blaze_util::JoinPath(server_dir, "server.starttime");
  if (!blaze_util::WriteFile(start_time, start_time_file)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Cannot write start time in server dir " << server_dir << ": "
        << GetLastErrorString();
  }
}

// On Linux we use a combination of PID and start time to identify the server
// process. That is supposed to be unique unless one can start more processes
// than there are PIDs available within a single jiffy.
bool VerifyServerProcess(int pid, const string& output_base) {
  string start_time;
  if (!GetStartTime(ToString(pid), &start_time)) {
    // Cannot read PID file from /proc . Process died meantime, all is good. No
    // stale server is present.
    return false;
  }

  string recorded_start_time;
  bool file_present = blaze_util::ReadFile(
      blaze_util::JoinPath(output_base, "server/server.starttime"),
      &recorded_start_time);

  // If start time file got deleted, but PID file didn't, assume that this is an
  // old Blaze process that doesn't know how to write start time files yet.
  return !file_present || recorded_start_time == start_time;
}

// Not supported.
void ExcludePathFromBackup(const string &path) {
}

int32_t GetExplicitSystemLimit(const int resource) {
  return -1;
}

}  // namespace blaze
