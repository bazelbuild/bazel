// Copyright 2016 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/workspace_layout.h"

#include <assert.h>
#include <unistd.h>  // access

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"

namespace blaze {

using std::string;
using std::vector;

static const char kWorkspaceMarker[] = "WORKSPACE";

string WorkspaceLayout::GetOutputRoot() {
  return blaze::GetOutputRoot();
}

bool WorkspaceLayout::InWorkspace(const string &workspace) {
  return access(
      blaze_util::JoinPath(workspace, kWorkspaceMarker).c_str(), F_OK) == 0;
}

string WorkspaceLayout::GetWorkspace(const string &cwd) {
  assert(!cwd.empty());
  string workspace = cwd;

  do {
    if (InWorkspace(workspace)) {
      return workspace;
    }
    workspace = blaze_util::Dirname(workspace);
  } while (!workspace.empty() && workspace != "/");
  return "";
}

string WorkspaceLayout::RcBasename() {
  return ".bazelrc";
}

static string FindDepotBlazerc(const string& workspace) {
  // Package semantics are ignored here, but that's acceptable because
  // blaze.blazerc is a configuration file.
  vector<string> candidates;
  WorkspaceLayout::WorkspaceRcFileSearchPath(&candidates);
  for (const auto& candidate : candidates) {
    string blazerc = blaze_util::JoinPath(workspace, candidate);
    if (!access(blazerc.c_str(), R_OK)) {
      return blazerc;
    }
  }
  return "";
}

static string FindAlongsideBinaryBlazerc(const string& cwd,
                                         const string& arg0) {
  string path = arg0[0] == '/' ? arg0 : blaze_util::JoinPath(cwd, arg0);
  string base = blaze_util::Basename(arg0);
  string binary_blazerc_path = path + "." + base + "rc";
  if (!access(binary_blazerc_path.c_str(), R_OK)) {
    return binary_blazerc_path;
  }
  return "";
}

static string FindSystemWideBlazerc() {
  string path = "/etc/bazel.bazelrc";
  if (!access(path.c_str(), R_OK)) {
    return path;
  }
  return "";
}

void WorkspaceLayout::FindCandidateBlazercPaths(
    const string& workspace, const string& cwd, const vector<string>& args,
    std::vector<string>* result) {
  result->push_back(FindDepotBlazerc(workspace));
  result->push_back(FindAlongsideBinaryBlazerc(cwd, args[0]));
  result->push_back(FindSystemWideBlazerc());
}

void WorkspaceLayout::WorkspaceRcFileSearchPath(
    vector<string>* candidates) {
  candidates->push_back("tools/bazel.rc");
}

bool WorkspaceLayout::WorkspaceRelativizeRcFilePath(const string &workspace,
                                                    string *path_fragment) {
  // Strip off the "%workspace%/" prefix and prepend the true workspace path.
  // In theory this could use alternate search paths for blazerc files.
  path_fragment->assign(
      blaze_util::JoinPath(workspace,
                           path_fragment->substr(WorkspacePrefixLength)));
  return true;
}

}  // namespace blaze
