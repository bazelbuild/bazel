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
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Ensures the graph contains the targets in the directory's package, if any, and in the
 * non-excluded packages in its subdirectories, and all those targets' transitive dependencies,
 * after a successful evaluation.
 */
public class PrepareDepsOfTargetsUnderDirectoryFunction implements SkyFunction {
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    PrepareDepsOfTargetsUnderDirectoryKey argument =
        (PrepareDepsOfTargetsUnderDirectoryKey) skyKey.argument();
    FilteringPolicy filteringPolicy = argument.getFilteringPolicy();
    CollectPackagesUnderDirectoryValue collectPackagesUnderDirectoryValue =
        (CollectPackagesUnderDirectoryValue)
            env.getValue(CollectPackagesUnderDirectoryValue.key(argument.getRecursivePkgKey()));
    if (env.valuesMissing()) {
      return null;
    }
    Map<RootedPath, Boolean> subdirMap =
        collectPackagesUnderDirectoryValue.getSubdirectoryTransitivelyContainsPackages();
    List<SkyKey> subdirKeys = new ArrayList<>(subdirMap.size());
    RepositoryName repositoryName = argument.getRecursivePkgKey().getRepository();
    ImmutableSet<PathFragment> excludedPaths = argument.getRecursivePkgKey().getExcludedPaths();

    PathFragment baseDir = argument.getRecursivePkgKey().getRootedPath().getRelativePath();
    for (Map.Entry<RootedPath, Boolean> subdirEntry : subdirMap.entrySet()) {
      if (subdirEntry.getValue()) {
        // Keep in rough sync with the logic in RecursiveDirectoryTraversalFunction#visitDirectory.
        RootedPath subdir = subdirEntry.getKey();
        PathFragment subdirRelativePath = subdir.getRelativePath().relativeTo(baseDir);
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
            PathFragment.filterPathsStartingWith(excludedPaths, subdirRelativePath);

        subdirKeys.add(
            PrepareDepsOfTargetsUnderDirectoryValue.key(
                repositoryName,
                subdir,
                excludedSubdirectoriesBeneathThisSubdirectory,
                filteringPolicy));
      }
    }
    if (collectPackagesUnderDirectoryValue.isDirectoryPackage()) {
      PackageIdentifier packageIdentifier =
          PackageIdentifier.create(
              argument.getRecursivePkgKey().getRepository(),
              argument.getRecursivePkgKey().getRootedPath().getRelativePath());
      PackageValue pkgValue =
          (PackageValue)
              Preconditions.checkNotNull(
                  env.getValue(PackageValue.key(packageIdentifier)),
                  collectPackagesUnderDirectoryValue);
      loadTransitiveTargets(env, pkgValue.getPackage(), filteringPolicy, subdirKeys);
    } else {
      env.getValues(subdirKeys);
    }
    return env.valuesMissing() ? null : PrepareDepsOfTargetsUnderDirectoryValue.INSTANCE;
  }

  // The additionalKeysToRequest argument allows us to batch skyframe dependencies a little more
  // aggressively. Since the keys computed in this method are independent from any other keys, we
  // can request our keys together with any other keys that are needed, possibly avoiding a restart.
  private static void loadTransitiveTargets(
      Environment env,
      Package pkg,
      FilteringPolicy filteringPolicy,
      Iterable<SkyKey> additionalKeysToRequest) {
    ResolvedTargets<Target> packageTargets =
        TargetPatternResolverUtil.resolvePackageTargets(pkg, filteringPolicy);
    ImmutableList.Builder<SkyKey> builder = ImmutableList.builder();
    for (Target target : packageTargets.getTargets()) {
      builder.add(TransitiveTraversalValue.key(target.getLabel()));
    }
    builder.addAll(additionalKeysToRequest);
    ImmutableList<SkyKey> skyKeys = builder.build();
    env.getValuesOrThrow(skyKeys, NoSuchPackageException.class, NoSuchTargetException.class);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
