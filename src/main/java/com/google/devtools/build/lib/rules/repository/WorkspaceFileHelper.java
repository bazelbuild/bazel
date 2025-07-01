// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

/** A class to help dealing with WORKSPACE.bazel and WORKSAPCE file */
public class WorkspaceFileHelper {

  public static boolean isValidRepoRoot(Path directory) {
    // Keep in sync with //src/main/cpp/workspace_layout.h
    return directory.getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.WORKSPACE_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.REPO_FILE_NAME).exists();
  }

  public static boolean matchWorkspaceFileName(PathFragment name) {
    return name.equals(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME)
        || name.equals(LabelConstants.WORKSPACE_FILE_NAME);
  }

  private WorkspaceFileHelper() {}
}
