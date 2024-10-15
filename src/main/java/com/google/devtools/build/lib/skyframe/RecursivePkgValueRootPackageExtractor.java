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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.List;

/** Looks up {@link RecursivePkgValue}s of given roots in a {@link WalkableGraph}. */
public class RecursivePkgValueRootPackageExtractor implements RootPackageExtractor {

  @Override
  public void streamPackagesFromRoots(
      SafeBatchCallback<PackageIdentifier> results,
      WalkableGraph graph,
      List<Root> roots,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      IgnoredSubdirectories ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException, QueryException {
    IgnoredSubdirectories filteredIgnoredSubdirectories =
        ignoredSubdirectories.filterForDirectory(directory);

    for (Root root : roots) {
      RootedPath rootedPath = RootedPath.toRootedPath(root, directory);
      RecursivePkgValue lookup =
          (RecursivePkgValue)
              graph.getValue(
                  RecursivePkgValue.key(repository, rootedPath, filteredIgnoredSubdirectories));
      if (lookup == null) {
        // A null lookup should only happen during post-analysis queries which have access to
        // --universe_scope logic. For builds lookup should never be null because {@link
        // RecursivePkgFunction} handles all errors in a --keep_going build. In a --nokeep_going
        // build, we should never reach this part of the code.
        throw new QueryException(
            String.format(
                "Unable to load package '%s' because package is not in scope. Check that all"
                    + " target patterns in query expression are within the --universe_scope of this"
                    + " query.",
                rootedPath),
            Code.TARGET_NOT_IN_UNIVERSE_SCOPE);
      }
      ImmutableList.Builder<PackageIdentifier> packageIds = ImmutableList.builder();
      for (String packageName : lookup.getPackages().toList()) {
        // TODO(bazel-team): Make RecursivePkgValue return NestedSet<PathFragment> so this transform
        // is unnecessary.
        PathFragment packageNamePathFragment = PathFragment.create(packageName);
        if (!Iterables.any(excludedSubdirectories, packageNamePathFragment::startsWith)) {
          packageIds.add(PackageIdentifier.create(repository, packageNamePathFragment));
        }
      }
      results.process(packageIds.build());
    }
  }
}
