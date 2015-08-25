// Copyright 2015 Google Inc. All rights reserved.
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

#include <limits.h>
#include <pwd.h>
#include <string.h>  // strerror
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/queue.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <libprocstat.h>

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
    fprintf(stderr,
            "WARNING: couldn't get file system type information for '%s': %s\n",
            output_base.c_str(), strerror(errno));
    return;
  }

  if (strcmp(buf.f_fstypename, "nfs") == 0) {
    fprintf(stderr,
            "WARNING: Output base '%s' is on NFS. This may lead "
            "to surprising failures and undetermined behavior.\n",
            output_base.c_str());
  }
}

string GetSelfPath() {
  char buffer[PATH_MAX] = {};
  ssize_t bytes = readlink("/proc/curproc/file", buffer, sizeof(buffer));
  if (bytes == sizeof(buffer)) {
    // symlink contents truncated
    bytes = -1;
    errno = ENAMETOOLONG;
  }
  if (bytes == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error reading /proc/curproc/file");
  }
  buffer[bytes] = '\0';  // readlink does not NUL-terminate
  return string(buffer);
}

pid_t GetPeerProcessId(int socket) { return -1; }

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
  // Move ourself into a low priority CPU scheduling group if the
  // machine is configured appropriately.  Fail silently, because this
  // isn't available on all kernels.

  if (io_nice_level >= 0) {
    if (blaze_util::sys_ioprio_set(
            IOPRIO_WHO_PROCESS, getpid(),
            IOPRIO_PRIO_VALUE(IOPRIO_CLASS_BE, io_nice_level)) < 0) {
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "ioprio_set() with class %d and level %d failed", IOPRIO_CLASS_BE,
           io_nice_level);
    }
  }
}

string GetProcessCWD(int pid) {
  if (kill(pid, 0) < 0) return "";
  auto procstat = procstat_open_sysctl();
  unsigned int n;
  auto p = procstat_getprocs(procstat, KERN_PROC_PID, pid, &n);
  string cwd;
  if (p) {
    if (n != 1) {
      pdie(blaze_exit_code::INTERNAL_ERROR,
           "expected exactly one process from procstat_getprocs, got %d", n);
    }
    auto files = procstat_getfiles(procstat, p, false);
    filestat *entry;
    STAILQ_FOREACH(entry, files, next) {
      if (entry->fs_uflags & PS_FST_UFLAG_CDIR) {
        cwd = entry->fs_path;
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

string GetDefaultHostJavabase() {
  // if JAVA_HOME is defined, then use it as default.
  const char *javahome = getenv("JAVA_HOME");
  if (javahome != NULL) {
    return string(javahome);
  }
  return "/usr/local/openjdk8";
}

}  // namespace blaze
