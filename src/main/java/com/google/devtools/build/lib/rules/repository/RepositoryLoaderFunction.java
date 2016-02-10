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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.ExternalPackageFunction;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * Creates a local or remote repository and checks its WORKSPACE file.
 */
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

    SkyKey workspaceKey = ExternalPackageFunction.key(
        RootedPath.toRootedPath(repository.getPath(), new PathFragment("WORKSPACE")));
    PackageValue workspacePackage = (PackageValue) env.getValue(workspaceKey);
    if (workspacePackage == null) {
      return null;
    }

    RepositoryName workspaceName;
    try {
      String workspaceNameStr = workspacePackage.getPackage().getWorkspaceName();
      workspaceName = workspaceNameStr.isEmpty()
          ? RepositoryName.create("") : RepositoryName.create("@" + workspaceNameStr);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }

    if (!workspaceName.isDefault() && !nameFromRule.equals(workspaceName)) {
      Path workspacePath = repository.getPath().getRelative("WORKSPACE");
      env.getListener().handle(Event.warn(Location.fromFile(workspacePath),
          "Workspace name in " + workspacePath + " (" + workspaceName + ") does not match the "
              + "name given in the repository's definition (" + nameFromRule + "); this will "
              + "cause a build error in future versions."));
    }

    return new RepositoryValue(nameFromRule, repository);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
