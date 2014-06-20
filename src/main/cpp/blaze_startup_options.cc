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
#include "blaze_startup_options.h"

#include <sys/stat.h>
#include <unistd.h>

#include "blaze_exit_code.h"
#include "blaze_util.h"
#include "util/file.h"
#include "util/strings.h"

namespace blaze {

using std::vector;

struct StartupOptions {};

BlazeStartupOptions::BlazeStartupOptions() {
  Init();
}

BlazeStartupOptions::BlazeStartupOptions(const BlazeStartupOptions &rhs)
    : output_base(rhs.output_base),
      install_base(rhs.install_base),
      output_root(rhs.output_root),
      output_user_root(rhs.output_user_root),
      block_for_lock(rhs.block_for_lock),
      host_jvm_debug(rhs.host_jvm_debug),
      host_jvm_profile(rhs.host_jvm_profile),
      host_javabase(rhs.host_javabase),
      host_jvm_args(rhs.host_jvm_args),
      use_blaze64(rhs.use_blaze64),
      batch(rhs.batch),
      batch_cpu_scheduling(rhs.batch_cpu_scheduling),
      io_nice_level(rhs.io_nice_level),
      max_idle_secs(rhs.max_idle_secs),
      skyframe(rhs.skyframe),
      allow_configurable_attributes(rhs.allow_configurable_attributes),
      option_sources(rhs.option_sources) {
}

BlazeStartupOptions::~BlazeStartupOptions() {
}

BlazeStartupOptions& BlazeStartupOptions::operator=(
    const BlazeStartupOptions &rhs) {
  Copy(rhs, this);
  return *this;
}

void BlazeStartupOptions::AddExtraOptions(vector<string> *result) {
}

bool BlazeStartupOptions::ProcessArgExtra(
    const char *arg, const char *next_arg, const string &rcfile,
    const char **value) {
  return false;
}

void BlazeStartupOptions::CheckForReExecuteOptions(
    int argc, const char *argv[]) {
}

static const char kJvmDir32[] = "/usr/lib/jvm/default-java";
static const char kJvmDir64[] = "/usr/lib64/jvm/default-java";

string BlazeStartupOptions::GetDefaultHostJavabase() {
  // First check for Java in /usr/lib64/jvm.
  bool arch_is_64 = GetBlazeArchitecture() == k64Bit;
  if (arch_is_64 && access(kJvmDir64, X_OK) == 0) {
    return kJvmDir64;
  }

  // 32- or 64-bit might live in /usr/lib/jvm.
  if (access(kJvmDir32, X_OK) == 0) {
    char buf[256];
    ssize_t len = readlink(kJvmDir32, buf, sizeof(buf));
    string jvm_dir = "";
    if (len > 0) {
      jvm_dir = string(buf, len);
    } else if (errno == EINVAL) {
      // Not a symbolic link
      jvm_dir = kJvmDir32;
    }

    if (jvm_dir != "") {
      bool dir_is_64 = jvm_dir.find("64") != string::npos;
      if (dir_is_64 == arch_is_64) {
        return kJvmDir32;
      }
    }
  }

  // Check the PATH for java.
  string path(getenv("PATH"));
  if (path.empty()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Could not get PATH to find Java");
  }

  vector<string> pieces = blaze_util::Split(path, ':');
  for (auto piece : pieces) {
    if (piece.empty()) {
      piece = ".";
    }

    struct stat file_stat;
    string candidate = blaze_util::JoinPath(piece, "java");
    if (access(candidate.c_str(), X_OK) == 0 &&
        stat(candidate.c_str(), &file_stat) == 0 &&
        S_ISREG(file_stat.st_mode)) {
      // Structure is JAVABASE/bin/java.
      return blaze_util::Dirname(blaze_util::Dirname(candidate));
    }
  }

  die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "Could not find Java");
}

string BlazeStartupOptions::GetJvm() {
  string java_program = host_javabase + "/bin/java";
  string rt_jar = host_javabase + "/jre/lib/rt.jar";
  if (access(rt_jar.c_str(), R_OK) == -1 ||
      access(java_program.c_str(), X_OK) == -1) {
    if (errno == ENOENT) {
      fprintf(stderr, "Couldn't find JDK at '%s'.\n",
              host_javabase.c_str());
    } else {
      fprintf(stderr, "Couldn't access %s: %s\n",
              host_javabase.c_str(), strerror(errno));
    }
    exit(1);
  }
  return java_program;
}

BlazeStartupOptions::Architecture BlazeStartupOptions::GetBlazeArchitecture() {
  return strcmp(BLAZE_JAVA_CPU, "64") == 0 ? k64Bit : k32Bit;
}

void BlazeStartupOptions::AddJVMSpecificArguments(const string &host_javabase,
                                                  vector<string> *result) {
  AddJVMArchArguments(GetBlazeArchitecture() == k64Bit, result);
}

}  // namespace blaze
