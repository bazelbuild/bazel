// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * Ensures the graph contains the targets in the directory's package, if any, and in the
 * non-excluded packages in its subdirectories, and all those targets' transitive dependencies,
 * after a successful evaluation.
 */
public class PrepareDepsOfTargetsUnderDirectoryFunction implements SkyFunction {
  private final BlazeDirectories directories;

  PrepareDepsOfTargetsUnderDirectoryFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    PrepareDepsOfTargetsUnderDirectoryKey argument =
        (PrepareDepsOfTargetsUnderDirectoryKey) skyKey.argument();
    final FilteringPolicy filteringPolicy = argument.getFilteringPolicy();
    RecursivePkgKey recursivePkgKey = argument.getRecursivePkgKey();
    ProcessPackageDirectory processPackageDirectory =
        new ProcessPackageDirectory(
            directories,
            new ProcessPackageDirectory.SkyKeyTransformer() {
              @Override
              public SkyKey makeSkyKey(
                  RepositoryName repository,
                  RootedPath subdirectory,
                  ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
                return PrepareDepsOfTargetsUnderDirectoryValue.key(
                    repository,
                    subdirectory,
                    excludedSubdirectoriesBeneathSubdirectory,
                    filteringPolicy);
              }
            });
    ProcessPackageDirectoryResult packageExistenceAndSubdirDeps =
        processPackageDirectory.getPackageExistenceAndSubdirDeps(
            recursivePkgKey.getRootedPath(),
            recursivePkgKey.getRepository(),
            env,
            recursivePkgKey.getExcludedPaths());
    if (env.valuesMissing()) {
      return null;
    }
    Iterable<SkyKey> keysToRequest = packageExistenceAndSubdirDeps.getChildDeps();
    if (packageExistenceAndSubdirDeps.packageExists()) {
      keysToRequest =
          Iterables.concat(
              ImmutableList.of(
                  CollectTargetsInPackageValue.key(
                      PackageIdentifier.create(
                          recursivePkgKey.getRepository(),
                          recursivePkgKey.getRootedPath().getRelativePath()),
                      filteringPolicy)),
              keysToRequest);
    }
    env.getValuesOrThrow(keysToRequest, NoSuchPackageException.class);
    if (env.valuesMissing()) {
      return null;
    }
    return PrepareDepsOfTargetsUnderDirectoryValue.INSTANCE;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
