// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Computes RefreshRootsValue by the global refresh() function call in WORKSPACE file header.
 * {@see WorkspaceFactory#refresh}
 */
public class RefreshRootsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    RootedPath workspacePath = ExternalPackageUtil.getWorkspacePath(env);
    if (env.valuesMissing()) {
      return null;
    }
    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue workspaceFileValue = (WorkspaceFileValue) env.getValue(workspaceKey);
    if (workspaceFileValue == null) {
      return null;
    }
    Package externalPackage = workspaceFileValue.getPackage();
    if (externalPackage.containsErrors()) {
      Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
      throw new ExternalPackageException(
          new BuildFileContainsErrorsException(
              LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    Map<String, RepositoryName> map = externalPackage.getRefreshRootsToRepository();

    TreeMap<PathFragment, RepositoryName> asRootsMap = Maps.newTreeMap();
    map.forEach((key, value) -> asRootsMap.put(PathFragment.create(key), value));
    return new RefreshRootsValue(asRootsMap);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
