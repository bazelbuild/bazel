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
#include "src/main/cpp/blaze_startup_options.h"

#include <assert.h>
#include <errno.h>  // errno, ENOENT
#include <stdlib.h>  // getenv, exit
#include <unistd.h>  // access

#include <cstdio>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using std::vector;

struct StartupOptions {};

string BlazeStartupOptions::GetProductName() {
  return "Bazel";
}

string BlazeStartupOptions::GetOutputRoot() {
  return blaze::GetOutputRoot();
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

blaze_exit_code::ExitCode BlazeStartupOptions::ProcessArgExtra(
    const char *arg, const char *next_arg, const string &rcfile,
    const char **value, bool *is_processed, string *error) {
  *is_processed = false;
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode BlazeStartupOptions::CheckForReExecuteOptions(
      int argc, const char *argv[], string *error) {
  return blaze_exit_code::SUCCESS;
}

string BlazeStartupOptions::GetDefaultHostJavabase() const {
  return blaze::GetDefaultHostJavabase();
}

string BlazeStartupOptions::GetJvm() {
  string java_program = GetHostJavabase() + "/bin/java";
  if (access(java_program.c_str(), X_OK) == -1) {
    if (errno == ENOENT) {
      fprintf(stderr, "Couldn't find java at '%s'.\n", java_program.c_str());
    } else {
      fprintf(stderr, "Couldn't access %s: %s\n", java_program.c_str(),
          strerror(errno));
    }
    exit(1);
  }
  // If the full JDK is installed
  string jdk_rt_jar = GetHostJavabase() + "/jre/lib/rt.jar";
  // If just the JRE is installed
  string jre_rt_jar = GetHostJavabase() + "/lib/rt.jar";
  if ((access(jdk_rt_jar.c_str(), R_OK) == 0)
      || (access(jre_rt_jar.c_str(), R_OK) == 0)) {
    return java_program;
  }
  fprintf(stderr, "Problem with java installation: "
      "couldn't find/access rt.jar in %s\n", GetHostJavabase().c_str());
  exit(1);
}

blaze_exit_code::ExitCode BlazeStartupOptions::AddJVMArguments(
    const string &host_javabase, vector<string> *result,
    const vector<string> &user_options, string *error) const {
  // Configure logging
  const string propFile = output_base + "/javalog.properties";
  if (!WriteFile(
      "handlers=java.util.logging.FileHandler\n"
      ".level=INFO\n"
      "java.util.logging.FileHandler.level=INFO\n"
      "java.util.logging.FileHandler.pattern="
      + output_base + "/java.log\n"
      "java.util.logging.FileHandler.limit=50000\n"
      "java.util.logging.FileHandler.count=1\n"
      "java.util.logging.FileHandler.formatter="
      "java.util.logging.SimpleFormatter\n",
      propFile)) {
    perror(("Couldn't write logging file " + propFile).c_str());
  } else {
    result->push_back("-Djava.util.logging.config.file=" + propFile);
  }
  return blaze_exit_code::SUCCESS;
}

string BlazeStartupOptions::RcBasename() {
  return ".bazelrc";
}

void BlazeStartupOptions::WorkspaceRcFileSearchPath(
    vector<string>* candidates) {
  candidates->push_back("tools/bazel.rc");
}

string BlazeStartupOptions::SystemWideRcPath() {
  return "/etc/bazel.bazelrc";
}

}  // namespace blaze
