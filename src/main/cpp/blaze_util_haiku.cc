#include <spawn.h>
#include <string.h>

#include <FindDirectory.h>
#include <Path.h>

#include <image.h>
#include <fs_info.h>
#include <OS.h>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using std::string;

string GetOutputRoot() {
  BPath path;
  if (find_directory(B_USER_CACHE_DIRECTORY, &path) == B_OK) {
    return blaze_util::JoinPath(path.Path(), "bazel");
  } else {
    return "/tmp";
  }
}

void WarnFilesystemType(const blaze_util::Path &output_base) {
  dev_t dev;
  fs_info info;
  int err;

  if ((err = dev = dev_for_path(output_base.AsNativePath().c_str())) < B_OK ||
      (err = fs_stat_dev(dev, &info)) < B_OK) {
    BAZEL_LOG(WARNING) << "couldn't get file system type information for '"
                       << output_base.AsPrintablePath()
                       << "': " << strerror(err);
    return;
  }

  // What about netfs, or FUSE/userlandfs?
  if (strcmp(info.fsh_name, "nfs") == 0 || strcmp(info.fsh_name, "nfs4") == 0) {
    BAZEL_LOG(WARNING) << "Output base '" << output_base.AsPrintablePath()
                       << "' is on NFS. This may lead to surprising failures "
                          "and undetermined behavior.";
  }
}

string GetSelfPath(const char* argv0) {
  int32 cookie = 0;
  image_info info;

  while (get_next_image_info(0, &cookie, &info) == B_OK)
    if (info.type == B_APP_IMAGE)
      return string(info.name);

  // Should never happen.
  BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
      << "Unable to determine the location of this Bazel executable.";
  return "";
}

uint64_t GetMillisecondsMonotonic() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000LL + (ts.tv_nsec / 1000000LL);
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // Stub
}

std::unique_ptr<blaze_util::Path> GetProcessCWD(int pid) {
  // I don't think this is possible on Haiku.
  return nullptr;
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

  return "/system/lib/openjdk11";
}

int ConfigureDaemonProcess(posix_spawnattr_t *attrp,
                           const StartupOptions &options) {
  // No interesting platform-specific details to configure on this platform.
  return 0;
}

void WriteSystemSpecificProcessIdentifier(const blaze_util::Path &server_dir,
                                          pid_t server_pid) {}

bool VerifyServerProcess(int pid, const blaze_util::Path &output_base) {
  // TODO: This only checks for the process's existence, not whether
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
