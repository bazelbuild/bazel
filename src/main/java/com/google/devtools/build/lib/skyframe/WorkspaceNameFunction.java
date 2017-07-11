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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} for {@link WorkspaceNameValue}s.
 *
 * <p>All errors (e.g. parsing errors or a symlink cycle encountered when consuming the WORKSPACE
 * file) result in a {@link NoSuchPackageException}.
 */
public class WorkspaceNameFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, WorkspaceNameFunctionException {
    boolean firstChunk = (boolean) skyKey.argument();
    Package externalPackage = firstChunk ? firstChunk(env) : parseFullPackage(env);
    if (externalPackage == null) {
      return null;
    }
    if (externalPackage.containsErrors()) {
      Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
      throw new WorkspaceNameFunctionException();
    }
    return WorkspaceNameValue.withName(externalPackage.getWorkspaceName());
  }

  /**
   * This just examines the first chunk of the WORKSPACE file to avoid circular dependencies and
   * overeager invalidation during package loading.
   */
  private Package firstChunk(Environment env) throws InterruptedException {
    SkyKey packageLookupKey = PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      return null;
    }
    RootedPath workspacePath = packageLookupValue.getRootedPath(Label.EXTERNAL_PACKAGE_IDENTIFIER);

    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(workspaceKey);
    if (value == null) {
      return null;
    }
    return value.getPackage();
  }

  private Package parseFullPackage(Environment env) throws InterruptedException {
    SkyKey externalPackageKey = PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageValue externalPackageValue = (PackageValue) env.getValue(externalPackageKey);
    if (externalPackageValue == null) {
      return null;
    }
    return externalPackageValue.getPackage();
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private class WorkspaceNameFunctionException extends SkyFunctionException {
    WorkspaceNameFunctionException() {
      super(new BuildFileContainsErrorsException(Label.EXTERNAL_PACKAGE_IDENTIFIER),
          Transience.PERSISTENT);
    }
  }
}
