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
#include <spawn.h>
#include <stdlib.h>
#include <string.h>  // strerror
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/queue.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/un.h>
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

void WarnFilesystemType(const blaze_util::Path &output_base) {
  struct statfs buf = {};
  if (statfs(output_base.AsNativePath().c_str(), &buf) < 0) {
    BAZEL_LOG(WARNING) << "couldn't get file system type information for '"
                       << output_base.AsPrintablePath()
                       << "': " << strerror(errno);
    return;
  }

  if (strcmp(buf.f_fstypename, "nfs") == 0) {
    BAZEL_LOG(WARNING) << "Output base '" << output_base.AsPrintablePath()
                       << "' is on NFS. This may lead to surprising failures "
                          "and undetermined behavior.";
  }
}

// OpenBSD does not provide an API for a running process to find the path of
// its own executable, so we try to figure out the path by inspecting argv[0].
// In theory this is inadequate, since the parent process can set argv[0] to
// anything, but in practice this is good enough.
string GetSelfPath(const string& argv0) {
  // TODO(aldersondrive): Add a new --bazel_executable_path startup option
  // only on platforms that need it), and inspect it here. If it's set, use its
  // value instead of applying the heuristics below.

  // If argv[0] starts with a slash, it's an absolute path. Use it.
  if (argv0.length() > 0 && argv0[0] == '/') {
    return argv0;
  }

  // Otherwise, if argv[0] contains a slash, then it's a relative path. Prepend
  // the current directory to form an absolute path.
  if (argv0.length() > 0 && argv0.find('/') != string::npos) {
    char buf[PATH_MAX];
    if (getcwd(buf, sizeof(buf)) == nullptr) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR) << "getcwd failed";
    }
    return string(buf) + "/" + argv0;
  }

  // TODO(aldersondrive): Try to find the executable by inspecting the PATH.

  // None of the above worked. Give up.
  BAZEL_DIE(blaze_exit_code::BAD_ARGV)
      << "Unable to determine the location of this Bazel executable. "
         "Currently, argv[0] must be an absolute or relative path to the "
         "executable.";
  return "";  // Never executed. Needed so compiler does not complain.
}

uint64_t GetMillisecondsMonotonic() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000LL + (ts.tv_nsec / 1000000LL);
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // Stubbed out so we can compile for OpenBSD.
}

blaze_util::Path GetProcessCWD(int pid) {
  // OpenBSD does not support looking up the working directory of another
  // process.
  return blaze_util::Path("");
}

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".so");
}

string GetSystemJavabase() {
  // If JAVA_HOME is defined, then use it as default.
  string javahome = GetPathEnv("JAVA_HOME");

  if (!javahome.empty()) {
    string javac = blaze_util::JoinPath(javahome, "bin/javac");
    if (access(javac.c_str(), X_OK) == 0) {
      return javahome;
    }
    BAZEL_LOG(WARNING)
        << "Ignoring JAVA_HOME, because it must point to a JDK, not a JRE.";
  }

  return "/usr/local/jdk-1.8.0";
}

int ConfigureDaemonProcess(posix_spawnattr_t *attrp,
                           const StartupOptions &options) {
  // No interesting platform-specific details to configure on this platform.
  return 0;
}

void WriteSystemSpecificProcessIdentifier(const blaze_util::Path &server_dir,
                                          pid_t server_pid) {}

bool VerifyServerProcess(int pid, const blaze_util::Path &output_base) {
  // TODO(lberki): This only checks for the process's existence, not whether
  // its start time matches. Therefore this might accidentally kill an
  // unrelated process if the server died and the PID got reused.
  return killpg(pid, 0) == 0;
}

// Not supported.
void ExcludePathFromBackup(const blaze_util::Path &path) {}

int32_t GetExplicitSystemLimit(const int resource) {
  return -1;
}

}  // namespace blaze
