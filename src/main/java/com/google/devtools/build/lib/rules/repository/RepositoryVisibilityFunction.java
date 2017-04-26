// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * Used for repositories that are declared in other repositories' WORKSPACE files.
 */
public class RepositoryVisibilityFunction implements SkyFunction {
  public static SkyKey key(RepositoryName parent, RepositoryName child) {
    return SkyKey.create(SkyFunctions.REPOSITORY_DEPENDENCY, Key.create(parent, child));
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    Key key = (Key) skyKey.argument();
    RepositoryDirectoryValue parentDir = (RepositoryDirectoryValue) env.getValue(
        RepositoryDirectoryValue.key(key.parent()));
    if (env.valuesMissing()) {
      return null;
    }

    SkyKey parentWorkspaceKey;
    if (parentDir.repositoryExists()) {
      parentWorkspaceKey = WorkspaceFileValue.key(
          RootedPath.toRootedPath(parentDir.getPath(), Label.EXTERNAL_PACKAGE_FILE_NAME));
    } else {
      // This is kind of hacky: the main repository won't exist under output_base/external, so RDF
      // won't find it above. However, we know that the parent repository has to exist, so we'll
      // just assume it's the main repository if RDF couldn't find it.
      PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(
          PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER));
      if (env.valuesMissing()) {
        return null;
      }
      Preconditions.checkState(packageLookupValue.packageExists());
      parentWorkspaceKey = WorkspaceFileValue.key(
          RootedPath.toRootedPath(packageLookupValue.getRoot(), Label.EXTERNAL_PACKAGE_FILE_NAME));
    }

    // This just looks for the child repo name. It doesn't care if the name is used for a different
    // repository (either a different type or a different path/url/commit/whichever) than is
    // actually being used. For example, if someone has a copy of Boost on their system that
    // they're using but a library they're depending downloads Boost, we just want "everyone" to
    // use the local copy of Boost, not error out. This is the check that the library actually
    // declares a repository "dependency" on @boost.
    while (true) {
      WorkspaceFileValue parentWorkspace = (WorkspaceFileValue) env.getValue(parentWorkspaceKey);
      if (env.valuesMissing()) {
        return null;
      }

      Rule child = parentWorkspace.getPackage().getRule(key.child().strippedName());
      if (child == null) {
        if (parentWorkspace.hasNext()) {
          parentWorkspaceKey = parentWorkspace.next();
        } else {
          break;
        }
      } else {
        return RepositoryVisibilityValue.OK;
      }
    }
    return RepositoryVisibilityValue.NOT_FOUND;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Represents a parent repository and a child repository that we expect to be defined in the
   * parent's WORKSPACE file.
   */
  @AutoValue
  public abstract static class Key {
    public static Key create(RepositoryName parent, RepositoryName child) {
      return new AutoValue_RepositoryVisibilityFunction_Key(parent, child);
    }

    public abstract RepositoryName parent();
    public abstract RepositoryName child();
  }

  /**
   * Returns if the repository definition was found or not.
   */
  public static class RepositoryVisibilityValue implements SkyValue {
    private static RepositoryVisibilityValue OK = new RepositoryVisibilityValue();
    private static RepositoryVisibilityValue NOT_FOUND = new RepositoryVisibilityValue();

    private RepositoryVisibilityValue() {
    }

    public boolean ok() {
      return this == OK;
    }
  }
}
