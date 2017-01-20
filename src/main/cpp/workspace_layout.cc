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

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"

namespace blaze {

using std::string;
using std::vector;

static const char kWorkspaceMarker[] = "WORKSPACE";

string WorkspaceLayout::GetOutputRoot() const {
  return blaze::GetOutputRoot();
}

bool WorkspaceLayout::InWorkspace(const string &workspace) const {
  return blaze_util::PathExists(
      blaze_util::JoinPath(workspace, kWorkspaceMarker));
}

string WorkspaceLayout::GetWorkspace(const string &cwd) const {
  assert(!cwd.empty());
  string workspace = cwd;

  do {
    if (InWorkspace(workspace)) {
      return workspace;
    }
    workspace = blaze_util::Dirname(workspace);
  } while (!workspace.empty() && !blaze_util::IsRootDirectory(workspace));
  return "";
}

static string FindDepotBlazerc(const blaze::WorkspaceLayout* workspace_layout,
                               const string& workspace) {
  // Package semantics are ignored here, but that's acceptable because
  // blaze.blazerc is a configuration file.
  vector<string> candidates;
  workspace_layout->WorkspaceRcFileSearchPath(&candidates);
  for (const auto& candidate : candidates) {
    string blazerc = blaze_util::JoinPath(workspace, candidate);
    if (blaze_util::CanReadFile(blazerc)) {
      return blazerc;
    }
  }
  return "";
}

static string FindAlongsideBinaryBlazerc(const string& cwd,
                                         const string& path_to_binary) {
  // TODO(b/32115171): This doesn't work on Windows. Fix this together with the
  // associated bug.
  const string path = blaze_util::IsAbsolute(path_to_binary)
                          ? path_to_binary
                          : blaze_util::JoinPath(cwd, path_to_binary);
  const string base = blaze_util::Basename(path_to_binary);
  const string binary_blazerc_path = path + "." + base + "rc";
  if (blaze_util::CanReadFile(binary_blazerc_path)) {
    return binary_blazerc_path;
  }
  return "";
}

void WorkspaceLayout::FindCandidateBlazercPaths(
    const string& workspace,
    const string& cwd,
    const string& path_to_binary,
    const vector<string>& startup_args,
    std::vector<string>* result) const {
  result->push_back(FindDepotBlazerc(this, workspace));
  result->push_back(FindAlongsideBinaryBlazerc(cwd, path_to_binary));
  result->push_back(FindSystemWideBlazerc());
}

void WorkspaceLayout::WorkspaceRcFileSearchPath(
    vector<string>* candidates) const {
  candidates->push_back("tools/bazel.rc");
}

bool WorkspaceLayout::WorkspaceRelativizeRcFilePath(const string &workspace,
                                                    string *path_fragment)
    const {
  // Strip off the "%workspace%/" prefix and prepend the true workspace path.
  // In theory this could use alternate search paths for blazerc files.
  path_fragment->assign(
      blaze_util::JoinPath(workspace,
                           path_fragment->substr(WorkspacePrefixLength)));
  return true;
}

}  // namespace blaze
