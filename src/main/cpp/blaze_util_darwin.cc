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

#include <libproc.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>

#include "blaze_exit_code.h"
#include "blaze_util.h"
#include "blaze_util_platform.h"
#include "util/file.h"
#include "util/strings.h"

namespace blaze {

using std::string;

string GetOutputRoot() {
  return "/var/tmp";
}

void WarnFilesystemType(const string& output_base) {
  // TODO(bazel-team): Should check for NFS.
  // TODO(bazel-team): Should check for case insensitive file systems?
}

pid_t GetPeerProcessId(int socket) {
  pid_t pid = 0;
  socklen_t len = sizeof(pid_t);
  if (getsockopt(socket, SOL_LOCAL, LOCAL_PEERPID, &pid, &len) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "can't get server pid from connection");
  }
  return pid;
}

string GetSelfPath() {
  char pathbuf[PROC_PIDPATHINFO_MAXSIZE] = {};
  int len = proc_pidpath(getpid(), pathbuf, sizeof(pathbuf));
  if (len == 0) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error calling proc_pidpath");
  }
  return string(pathbuf, len);
}

uint64 MonotonicClock() {
  struct timeval ts = {};
  if (gettimeofday(&ts, NULL) < 0) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error calling gettimeofday");
  }
  return ts.tv_sec * 1000000000LL + ts.tv_usec * 1000;
}

uint64 ProcessClock() {
  return clock() * (1000000000LL / CLOCKS_PER_SEC);
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // stubbed out so we can compile for Darwin.
}

string GetProcessCWD(int pid) {
  struct proc_vnodepathinfo info = {};
  if (proc_pidinfo(
          pid, PROC_PIDVNODEPATHINFO, 0, &info, sizeof(info)) != sizeof(info)) {
    return "";
  }
  return string(info.pvi_cdir.vip_path);
}

bool IsSharedLibrary(string filename) {
  return blaze_util::ends_with(filename, ".dylib");
}

string GetDefaultHostJavabase() {
  const char *java_home = getenv("JAVA_HOME");
  if (java_home) {
    return std::string(java_home);
  }

  FILE *output = popen("/usr/libexec/java_home -v 1.7+", "r");
  if (output == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "Could not run /usr/libexec/java_home");
  }

  char buf[512];
  char *result = fgets(buf, sizeof(buf), output);
  pclose(output);
  if (result == NULL) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "No output from /usr/libexec/java_home");
  }

  string javabase = buf;
  if (javabase.empty()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Empty output from /usr/libexec/java_home - "
        "install a JDK, or install a JRE and point your JAVA_HOME to it");
  }

  // The output ends with a \n, trim it off.
  return javabase.substr(0, javabase.length()-1);
}

}   // namespace blaze.
