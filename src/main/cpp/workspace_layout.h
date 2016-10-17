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
#ifndef BAZEL_SRC_MAIN_CPP_WORKSPACE_LAYOUT_H_
#define BAZEL_SRC_MAIN_CPP_WORKSPACE_LAYOUT_H_

#include <string>
#include <vector>

namespace blaze {

// Provides methods to compute paths related to the workspace.
//
// All methods in this class ought to be static because we reference them as
// if they were global, and we do so from very early startup stages.
//
// The reason this is a class and not a namespace is because of historical
// reasons, as this was split out from the BlazeStartupOptions class.
// TODO(bazel-team): Reconsider dropping the class in favor of free functions.
class WorkspaceLayout {
 public:
  WorkspaceLayout() = delete;

  // Returns the directory to use for storing outputs.
  static std::string GetOutputRoot();

  // Given the working directory, returns the nearest enclosing directory with a
  // WORKSPACE file in it.  If there is no such enclosing directory, returns "".
  //
  // E.g., if there was a WORKSPACE file in foo/bar/build_root:
  // GetWorkspace('foo/bar') --> ''
  // GetWorkspace('foo/bar/build_root') --> 'foo/bar/build_root'
  // GetWorkspace('foo/bar/build_root/biz') --> 'foo/bar/build_root'
  //
  // The returned path is relative or absolute depending on whether cwd was
  // relative or absolute.
  static std::string GetWorkspace(const std::string& cwd);

  // Returns if workspace is a valid build workspace.
  static bool InWorkspace(const std::string& workspace);

  // Returns the basename for the rc file.
  static std::string RcBasename();

  // Returns the candidate pathnames for the RC files.
  static void FindCandidateBlazercPaths(const std::string& workspace,
                                        const std::string& cwd,
                                        const std::vector<std::string>& args,
                                        std::vector<std::string>* result);

  // Returns the candidate pathnames for the RC file in the workspace,
  // the first readable one of which will be chosen.
  // It is ok if no usable candidate exists.
  static void WorkspaceRcFileSearchPath(std::vector<std::string>* candidates);

  // Turn a %workspace%-relative import into its true name in the filesystem.
  // path_fragment is modified in place.
  // Unlike WorkspaceRcFileSearchPath, it is an error if no import file exists.
  static bool WorkspaceRelativizeRcFilePath(const std::string& workspace,
                                            std::string* path_fragment);

  static constexpr char WorkspacePrefix[] = "%workspace%/";
  static const int WorkspacePrefixLength = sizeof WorkspacePrefix - 1;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_WORKSPACE_LAYOUT_H_
