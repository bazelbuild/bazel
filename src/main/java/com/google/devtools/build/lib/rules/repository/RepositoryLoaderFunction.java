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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import javax.annotation.Nullable;

/** Creates a local or remote repository. */
public class RepositoryLoaderFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    // This cannot be combined with {@link RepositoryDelegatorFunction}. RDF fetches the
    // repository and must not have a Skyframe restart after writing it (otherwise the repository
    // would be re-downloaded).
    RepositoryName nameFromRule = (RepositoryName) skyKey.argument();
    SkyKey repositoryKey = RepositoryDirectoryValue.key(nameFromRule);
    RepositoryDirectoryValue repository = (RepositoryDirectoryValue) env.getValue(repositoryKey);
    if (repository == null) {
      return null;
    }
    if (!repository.repositoryExists()) {
      return RepositoryValue.notFound(nameFromRule);
    }
    RootedPath workspaceFilePath;
    try {
      workspaceFilePath =
          WorkspaceFileHelper.getWorkspaceRootedFile(Root.fromPath(repository.getPath()), env);
      if (workspaceFilePath == null) {
        return null;
      }
    } catch (IOException e) {
      throw new RepositoryLoaderFunctionException(
          new IOException(
              "Could not determine workspace file (\"WORKSPACE.bazel\" or \"WORKSPACE\"): "
                  + e.getMessage()),
          Transience.PERSISTENT);
    }
    SkyKey workspaceKey = WorkspaceFileValue.key(workspaceFilePath);
    WorkspaceFileValue workspacePackage = (WorkspaceFileValue) env.getValue(workspaceKey);
    if (workspacePackage == null) {
      return null;
    }

    try {
      String workspaceNameStr = workspacePackage.getPackage().getWorkspaceName();
      if (workspaceNameStr.isEmpty()) {
        RepositoryName.create("");
      } else {
        RepositoryName.create("@" + workspaceNameStr);
      }
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }

    return RepositoryValue.success(nameFromRule, repository);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** An exception thrown by RepositoryLoaderFunction */
  public static class RepositoryLoaderFunctionException extends SkyFunctionException {

    /** Error reading or writing to the filesystem. */
    public RepositoryLoaderFunctionException(IOException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
