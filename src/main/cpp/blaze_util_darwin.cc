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

#include "src/main/cpp/blaze_util_platform.h"

#include <sys/types.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <libproc.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <CoreFoundation/CoreFoundation.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;
using std::string;
using std::vector;

// A stack based class for RAII type handling of CF based types that need
// CFRelease called on them. Checks for NULL before calling release.
template <typename T> class CFScopedReleaser {
 public:
  explicit CFScopedReleaser(T value) : value_(value) { }
  ~CFScopedReleaser() {
    if (isValid()) {
      CFRelease(value_);
    }
  }
  T get() { return value_; }
  operator T() { return value_; }
  bool isValid() { return value_ != NULL; }

 private:
  T value_;

  CFScopedReleaser() { }
  CFScopedReleaser(const CFScopedReleaser&);
  CFScopedReleaser& operator=(CFScopedReleaser&);
};

// Convert a CFStringRef to a UTF8 encoded c string
static string UTF8StringFromCFStringRef(CFStringRef cf_string) {
  std::string utf8_string;
  if (cf_string) {
    CFIndex length = CFStringGetLength(cf_string);
    CFIndex max_size =
        CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
    vector<char> buffer(max_size);
    if (CFStringGetCString(cf_string, &buffer[0], max_size,
                           kCFStringEncodingUTF8)) {
      utf8_string = &buffer[0];
    }
  }
  return utf8_string;
}

// Extract description from a CFError.
static string DescriptionFromCFError(CFErrorRef cf_err) {
  if (!cf_err) {
    return "";
  }
  CFScopedReleaser<CFStringRef> cf_err_string(CFErrorCopyDescription(cf_err));
  return UTF8StringFromCFStringRef(cf_err_string);
}

string GetOutputRoot() {
  return "/var/tmp";
}

void WarnFilesystemType(const string& output_base) {
  // Check to see if we are on a non-local drive.
  CFScopedReleaser<CFURLRef> cf_url(CFURLCreateFromFileSystemRepresentation(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(output_base.c_str()),
      output_base.length(), true));
  CFBooleanRef cf_local = NULL;
  CFErrorRef cf_error = NULL;
  if (!cf_url.isValid() ||
      !CFURLCopyResourcePropertyForKey(cf_url, kCFURLVolumeIsLocalKey,
                                       &cf_local, &cf_error)) {
    CFScopedReleaser<CFErrorRef> cf_error_releaser(cf_error);
    string error_desc = DescriptionFromCFError(cf_error_releaser);
    fprintf(stderr, "Warning: couldn't get file system type information for "
            "'%s'", output_base.c_str());
    if (error_desc.length() > 0) {
      fprintf(stderr, " - '%s'", error_desc.c_str());
    }
    fprintf(stderr, "\n");
    return;
  }
  CFScopedReleaser<CFBooleanRef> cf_local_releaser(cf_local);
  if (!CFBooleanGetValue(cf_local_releaser)) {
    fprintf(stderr, "Warning: Output base '%s' is on a non-local drive. This "
            "may lead to surprising failures and undetermined behavior.\n",
            output_base.c_str());
  }
}

string GetSelfPath() {
  char pathbuf[PROC_PIDPATHINFO_MAXSIZE] = {};
  int len = proc_pidpath(getpid(), pathbuf, sizeof(pathbuf));
  if (len == 0) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error calling proc_pidpath");
  }
  return string(pathbuf, len);
}

uint64_t GetMillisecondsMonotonic() {
  struct timeval ts = {};
  if (gettimeofday(&ts, NULL) < 0) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "error calling gettimeofday");
  }
  return ts.tv_sec * 1000LL + ts.tv_usec / 1000LL;
}

uint64_t GetMillisecondsSinceProcessStart() {
  return (clock() * 1000LL) / CLOCKS_PER_SEC;
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

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".dylib");
}

string GetDefaultHostJavabase() {
  string java_home = GetEnv("JAVA_HOME");
  if (!java_home.empty()) {
    return java_home;
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

void WriteSystemSpecificProcessIdentifier(
    const string& server_dir, pid_t server_pid) {
}

bool VerifyServerProcess(int pid, const string &output_base) {
  // TODO(lberki): This only checks for the process's existence, not whether
  // its start time matches. Therefore this might accidentally kill an
  // unrelated process if the server died and the PID got reused.
  return killpg(pid, 0) == 0;
}

// Sets a flag on path to exclude the path from Apple's automatic backup service
// (Time Machine)
void ExcludePathFromBackup(const string &path) {
  CFScopedReleaser<CFURLRef> cf_url(CFURLCreateFromFileSystemRepresentation(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(path.c_str()),
      path.length(), true));
  if (!cf_url.isValid()) {
    fprintf(stderr, "Warning: unable to exclude '%s' from backups\n",
            path.c_str());
    return;
  }
  CFErrorRef cf_error = NULL;
  if (!CFURLSetResourcePropertyForKey(cf_url, kCFURLIsExcludedFromBackupKey,
                                      kCFBooleanTrue, &cf_error)) {
    CFScopedReleaser<CFErrorRef> cf_error_releaser(cf_error);
    string error_desc = DescriptionFromCFError(cf_error_releaser);
    fprintf(stderr, "Warning: unable to exclude '%s' from backups",
            path.c_str());
    if (error_desc.length() > 0) {
      fprintf(stderr, " - '%s'", error_desc.c_str());
    }
    fprintf(stderr, "\n");
    return;
  }
}

int32_t GetExplicitSystemLimit(const int resource) {
  const char* sysctl_name;
  switch (resource) {
    case RLIMIT_NOFILE:
      sysctl_name = "kern.maxfilesperproc";
      break;
    case RLIMIT_NPROC:
      sysctl_name = "kern.maxprocperuid";
      break;
    default:
      return 0;
  }

  int32_t limit;
  size_t len = sizeof(limit);
  if (sysctlbyname(sysctl_name, &limit, &len, nullptr, 0) == -1) {
    fprintf(stderr, "Warning: failed to get value of sysctl %s: %s\n",
            sysctl_name, std::strerror(errno));
    return 0;
  }
  if (len != sizeof(limit)) {
    fprintf(stderr, "Warning: failed to get value of sysctl %s: returned "
            "data length %zd did not match expected size %zd\n",
            sysctl_name, len, sizeof(limit));
    return 0;
  }
  return limit;
}

}   // namespace blaze.
