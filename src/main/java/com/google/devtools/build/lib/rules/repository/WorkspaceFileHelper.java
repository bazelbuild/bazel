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

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import javax.annotation.Nullable;

/** A class to help dealing with WORKSPACE.bazel and WORKSAPCE file */
public class WorkspaceFileHelper {

  /**
   * Get a RootedPath of the WORKSPACE file we should use for a given directory. This function
   * returns a RootedPath to <directory>/WORKSPACE.bazel file if it exists and it's a regular file
   * or file symlink, otherwise, return a RootedPath to <directory>/WORKSPACE. The caller of this
   * function cannot assume the path returned exists, because in the second case we don't check the
   * FileValue of `WORKSPACE` file.
   *
   * @param directory The directory that could contain WORKSPACE.bazel or WORKSPACE file.
   * @param env Skyframe env
   * @return A RootedPath to the WORKSPACE file we should use for the given directory.
   */
  @Nullable
  public static RootedPath getWorkspaceRootedFile(RootedPath directory, Environment env)
      throws IOException, InterruptedException {
    RootedPath workspaceRootedFile =
        RootedPath.toRootedPath(
            directory.getRoot(),
            directory
                .getRootRelativePath()
                .getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME));

    SkyKey workspaceFileKey = FileValue.key(workspaceRootedFile);
    FileValue value;
    value = (FileValue) env.getValueOrThrow(workspaceFileKey, IOException.class);
    if (value == null) {
      return null;
    }

    if (value.isFile() && !value.isSpecialFile()) {
      return workspaceRootedFile;
    }

    return RootedPath.toRootedPath(
        directory.getRoot(),
        directory.getRootRelativePath().getRelative(LabelConstants.WORKSPACE_FILE_NAME));
  }

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
