// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Skyframe implementation of {@link PackageRootResolver}.
 *
 * <p> Note: you should not use this class inside any SkyFunctions.
 */
public final class SkyframePackageRootResolver implements PackageRootResolver {
  private final SkyframeExecutor executor;
  private final ExtendedEventHandler eventHandler;

  public SkyframePackageRootResolver(SkyframeExecutor executor, ExtendedEventHandler eventHandler) {
    this.executor = executor;
    this.eventHandler = eventHandler;
  }

  @Override
  public Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
      throws PackageRootResolutionException, InterruptedException {
    return executor.getArtifactRootsForFiles(eventHandler, execPaths);
  }
  
  @Override
  @Nullable
  public Map<PathFragment, Root> findPackageRoots(Iterable<PathFragment> execPaths)
      throws PackageRootResolutionException, InterruptedException {
    return executor.getArtifactRoots(eventHandler, execPaths);
  }
}
