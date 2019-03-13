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

#include <errno.h>  // errno, ENAMETOOLONG
#include <limits.h>
#include <pwd.h>
#include <signal.h>
#include <string.h>  // strerror
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/queue.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <libprocstat.h>  // must be included after <sys/...> headers

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

string GetOutputRoot() {
  char buf[2048];
  struct passwd pwbuf;
  struct passwd *pw = NULL;
  int uid = getuid();
  int r = getpwuid_r(uid, &pwbuf, buf, 2048, &pw);
  if (r != -1 && pw != NULL) {
    return blaze_util::JoinPath(pw->pw_dir, ".cache/bazel");
  } else {
    return "/tmp";
  }
}

void WarnFilesystemType(const string &output_base) {
  struct statfs buf = {};
  if (statfs(output_base.c_str(), &buf) < 0) {
    BAZEL_LOG(WARNING) << "couldn't get file system type information for '"
                       << output_base << "': " << strerror(errno);
    return;
  }

  if (strcmp(buf.f_fstypename, "nfs") == 0) {
    BAZEL_LOG(WARNING) << "Output base '" << output_base
                       << "' is on NFS. This may lead to surprising failures "
                          "and undetermined behavior.";
  }
}

string GetSelfPath() {
  char buffer[PATH_MAX] = {};
  auto pid = getpid();
  if (kill(pid, 0) < 0) return "";
  auto procstat = procstat_open_sysctl();
  unsigned int n;
  auto p = procstat_getprocs(procstat, KERN_PROC_PID, pid, &n);
  if (p) {
    if (n != 1) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "expected exactly one process from procstat_getprocs, got " << n
          << ": " << GetLastErrorString();
    }
    auto r = procstat_getpathname(procstat, p, buffer, PATH_MAX);
    if (r != 0) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "procstat_getpathname failed: " << GetLastErrorString();
    }
    procstat_freeprocs(procstat, p);
  }
  procstat_close(procstat);
  return string(buffer);
}

uint64_t GetMillisecondsMonotonic() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000LL + (ts.tv_nsec / 1000000LL);
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // Stubbed out so we can compile for FreeBSD.
}

string GetProcessCWD(int pid) {
  if (kill(pid, 0) < 0) return "";
  auto procstat = procstat_open_sysctl();
  unsigned int n;
  auto p = procstat_getprocs(procstat, KERN_PROC_PID, pid, &n);
  string cwd;
  if (p) {
    if (n != 1) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "expected exactly one process from procstat_getprocs, got " << n
          << ": " << GetLastErrorString();
    }
    auto files = procstat_getfiles(procstat, p, false);
    filestat *entry;
    STAILQ_FOREACH(entry, files, next) {
      if (entry->fs_uflags & PS_FST_UFLAG_CDIR) {
        if (entry->fs_path) {
          cwd = entry->fs_path;
        } else {
          cwd = "";
        }
      }
    }
    procstat_freefiles(procstat, files);
    procstat_freeprocs(procstat, p);
  }
  procstat_close(procstat);
  return cwd;
}

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".so");
}

string GetSystemJavabase() {
  // if JAVA_HOME is defined, then use it as default.
  string javahome = GetPathEnv("JAVA_HOME");

  if (!javahome.empty()) {
    string javac = blaze_util::JoinPath(javahome, "bin/javac");
    if (access(javac.c_str(), X_OK) == 0) {
      return javahome;
    }
    BAZEL_LOG(WARNING)
        << "Ignoring JAVA_HOME, because it must point to a JDK, not a JRE.";
  }

  return "/usr/local/openjdk8";
}

void WriteSystemSpecificProcessIdentifier(
    const string& server_dir, pid_t server_pid) {
}

bool VerifyServerProcess(int pid, const string &output_base) {
  // TODO(lberki): This only checks for the process's existence, not whether
  // its start time matches. Therefore this might accidentally kill an
  // unrelated process if the server died and the PID got reused.
  return killpg(pid, 0) == 0;
}

// Not supported.
void ExcludePathFromBackup(const string &path) {
}

int32_t GetExplicitSystemLimit(const int resource) {
  return -1;
}

}  // namespace blaze
