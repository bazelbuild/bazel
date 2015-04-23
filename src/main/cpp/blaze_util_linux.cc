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

#include <limits.h>
#include <string.h>  // strerror
#include <sys/statfs.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
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

void WarnFilesystemType(const string& output_base) {
  struct statfs buf = {};
  if (statfs(output_base.c_str(), &buf) < 0) {
    fprintf(stderr,
            "WARNING: couldn't get file system type information for '%s': %s\n",
            output_base.c_str(), strerror(errno));
    return;
  }

  if (buf.f_type == 0x00006969) {  // NFS_SUPER_MAGIC
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

uint64 MonotonicClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

uint64 ProcessClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // Move ourself into a low priority CPU scheduling group if the
  // machine is configured appropriately.  Fail silently, because this
  // isn't available on all kernels.
  if (FILE *f = fopen("/dev/cgroup/cpu/batch/tasks", "w")) {
    fprintf(f, "%d", getpid());
    fclose(f);
  }

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
          ("/proc/" + std::to_string(pid) + "/cwd").c_str(),
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

}  // namespace blaze
