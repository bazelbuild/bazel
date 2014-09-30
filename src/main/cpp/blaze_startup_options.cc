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

#include <assert.h>
#include <errno.h>  // errno, ENOENT
#include <stdlib.h>  // getenv, exit
#include <unistd.h>  // access

#include <cstdio>

#include "blaze_exit_code.h"
#include "blaze_util_platform.h"
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
      host_jvm_args(rhs.host_jvm_args),
      use_blaze64(rhs.use_blaze64),
      batch(rhs.batch),
      batch_cpu_scheduling(rhs.batch_cpu_scheduling),
      io_nice_level(rhs.io_nice_level),
      max_idle_secs(rhs.max_idle_secs),
      skyframe(rhs.skyframe),
      watchfs(rhs.watchfs),
      allow_configurable_attributes(rhs.allow_configurable_attributes),
      option_sources(rhs.option_sources),
      host_javabase(rhs.host_javabase) {}

BlazeStartupOptions::~BlazeStartupOptions() {
}

BlazeStartupOptions& BlazeStartupOptions::operator=(
    const BlazeStartupOptions &rhs) {
  Copy(rhs, this);
  return *this;
}

string BlazeStartupOptions::GetOutputRoot() {
  return "/var/tmp";
}

void BlazeStartupOptions::AddExtraOptions(vector<string> *result) const {}

static const char kWorkspaceMarker[] = "WORKSPACE";

// static
bool BlazeStartupOptions::InWorkspace(const string &workspace) {
  return access(
      blaze_util::JoinPath(workspace, kWorkspaceMarker).c_str(), F_OK) == 0;
}

// static
string BlazeStartupOptions::GetWorkspace(const string &cwd) {
  assert(!cwd.empty());
  string workspace = cwd;

  do {
    if (access(blaze_util::JoinPath(
            workspace, kWorkspaceMarker).c_str(), F_OK) != -1) {
      return workspace;
    }
    workspace = blaze_util::Dirname(workspace);
  } while (!workspace.empty() && workspace != "/");
  return "";
}

bool BlazeStartupOptions::ProcessArgExtra(
    const char *arg, const char *next_arg, const string &rcfile,
    const char **value) {
  return false;
}

void BlazeStartupOptions::CheckForReExecuteOptions(
    int argc, const char *argv[]) {
}

string BlazeStartupOptions::GetDefaultHostJavabase() const {
  return blaze::GetDefaultHostJavabase();
}

string BlazeStartupOptions::GetJvm() {
  string java_program = GetHostJavabase() + "/bin/java";
  string rt_jar = GetHostJavabase() + "/jre/lib/rt.jar";
  if (access(rt_jar.c_str(), R_OK) == -1 ||
      access(java_program.c_str(), X_OK) == -1) {
    if (errno == ENOENT) {
      fprintf(stderr, "Couldn't find JDK at '%s'.\n",
              GetHostJavabase().c_str());
    } else {
      fprintf(stderr, "Couldn't access %s: %s\n", GetHostJavabase().c_str(),
              strerror(errno));
    }
    exit(1);
  }
  return java_program;
}

BlazeStartupOptions::Architecture BlazeStartupOptions::GetBlazeArchitecture()
    const {
  return strcmp(BLAZE_JAVA_CPU, "64") == 0 ? k64Bit : k32Bit;
}

void BlazeStartupOptions::AddJVMArguments(const string &host_javabase,
                                          vector<string> *result) const {
  // TODO(bazel-team): see what tuning options make sense in the
  // open-source world.
}

string BlazeStartupOptions::RcBasename() {
  return ".bazelrc";
}

void BlazeStartupOptions::WorkspaceRcFileSearchPath(
    vector<string>* candidates) {
  candidates->push_back("tools/bazel.rc");
}

}  // namespace blaze
