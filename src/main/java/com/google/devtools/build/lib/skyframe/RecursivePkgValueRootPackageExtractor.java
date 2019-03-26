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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.LinkedHashSet;
import java.util.List;

/** Looks up {@link RecursivePkgValue}s of given roots in a {@link WalkableGraph}. */
public class RecursivePkgValueRootPackageExtractor implements RootPackageExtractor {

  public Iterable<PathFragment> getPackagesFromRoots(
      WalkableGraph graph,
      List<Root> roots,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException {

    ImmutableSet filteredBlacklistedSubdirectories =
        ImmutableSet.copyOf(
            Iterables.filter(
                blacklistedSubdirectories,
                path -> !path.equals(directory) && path.startsWith(directory)));

    LinkedHashSet<PathFragment> packageNames = new LinkedHashSet<>();
    for (Root root : roots) {
      // Note: no need to check if lookup == null because it will never be null.
      // {@link RecursivePkgFunction} handles all errors in a keep_going build.
      // In a nokeep_going build, we would never reach this part of the code.
      RecursivePkgValue lookup =
          (RecursivePkgValue)
              graph.getValue(
                  RecursivePkgValue.key(
                      repository,
                      RootedPath.toRootedPath(root, directory),
                      filteredBlacklistedSubdirectories));
      Preconditions.checkState(
          lookup != null,
          "Root %s in repository %s could not be found in the graph.",
          root.asPath(),
          repository.getName());
      for (String packageName : lookup.getPackages()) {
        // TODO(bazel-team): Make RecursivePkgValue return NestedSet<PathFragment> so this transform
        // is unnecessary.
        PathFragment packageNamePathFragment = PathFragment.create(packageName);
        if (!Iterables.any(
            excludedSubdirectories,
            excludedSubdirectory -> packageNamePathFragment.startsWith(excludedSubdirectory))) {
          packageNames.add(packageNamePathFragment);
        }
      }
    }

    return packageNames;
  }
}
