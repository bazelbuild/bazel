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
package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;

public class WorkspaceFileHelper {

  public static RootedPath getWorkspaceRootedFile(Root directory) {
    return getWorkspaceRootedFile(RootedPath.toRootedPath(directory, PathFragment.EMPTY_FRAGMENT));
  }

  public static RootedPath getWorkspaceRootedFile(RootedPath directory) {
    RootedPath workspaceRootedFile =
        RootedPath.toRootedPath(
            directory.getRoot(),
            directory
                .getRootRelativePath()
                .getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME));

    if (!workspaceRootedFile.asPath().exists()) {
      workspaceRootedFile = RootedPath.toRootedPath(
          directory.getRoot(),
          directory
              .getRootRelativePath()
              .getRelative(LabelConstants.WORKSPACE_FILE_NAME));
    }
    return workspaceRootedFile;
  }

  public static boolean doesWorkspaceFileExistUnder(Path directory) {
    return directory.getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME).exists() ||
        directory.getRelative(LabelConstants.WORKSPACE_FILE_NAME).exists();
  }

  public static boolean matchWorkspaceFileName(String name) {
    return matchWorkspaceFileName(PathFragment.create(name));
  }

  public static boolean matchWorkspaceFileName(PathFragment name) {
    return name.equals(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME)
        || name.equals(LabelConstants.WORKSPACE_FILE_NAME);
  }

  public static boolean endsWithWorkspaceFileName(PathFragment pathFragment) {
    return pathFragment.endsWith(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME)
        || pathFragment.endsWith(LabelConstants.WORKSPACE_FILE_NAME);
  }
}
