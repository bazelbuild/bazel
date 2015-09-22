// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Ensures the graph contains the targets in the directory's package, if any, and in the
 * non-excluded packages in its subdirectories, and all those targets' transitive dependencies,
 * after a successful evaluation.
 *
 * <p>Computes {@link PrepareDepsOfTargetsUnderDirectoryValue} which describes whether the
 * directory is a package and how many non-excluded packages exist below each of the directory's
 * subdirectories.
 */
public class PrepareDepsOfTargetsUnderDirectoryFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    PrepareDepsOfTargetsUnderDirectoryKey argument =
        (PrepareDepsOfTargetsUnderDirectoryKey) skyKey.argument();
    FilteringPolicy filteringPolicy = argument.getFilteringPolicy();
    RecursivePkgKey recursivePkgKey = argument.getRecursivePkgKey();
    return new MyTraversalFunction(filteringPolicy).visitDirectory(recursivePkgKey, env);
  }

  private static class MyTraversalFunction
      extends RecursiveDirectoryTraversalFunction<MyVisitor,
      PrepareDepsOfTargetsUnderDirectoryValue> {

    private final FilteringPolicy filteringPolicy;

    private MyTraversalFunction(FilteringPolicy filteringPolicy) {
      this.filteringPolicy = filteringPolicy;
    }

    @Override
    protected PrepareDepsOfTargetsUnderDirectoryValue getEmptyReturn() {
      return PrepareDepsOfTargetsUnderDirectoryValue.EMPTY;
    }

    @Override
    protected MyVisitor getInitialVisitor() {
      return new MyVisitor(filteringPolicy);
    }

    @Override
    protected SkyKey getSkyKeyForSubdirectory(RepositoryName repository, RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
      return PrepareDepsOfTargetsUnderDirectoryValue.key(repository, subdirectory,
          excludedSubdirectoriesBeneathSubdirectory, filteringPolicy);
    }

    @Override
    protected PrepareDepsOfTargetsUnderDirectoryValue aggregateWithSubdirectorySkyValues(
        MyVisitor visitor, Map<SkyKey, SkyValue> subdirectorySkyValues) {
      // Aggregate the child subdirectory package state.
      ImmutableMap.Builder<RootedPath, Boolean> builder = ImmutableMap.builder();
      for (SkyKey key : subdirectorySkyValues.keySet()) {
        PrepareDepsOfTargetsUnderDirectoryKey prepDepsKey =
            (PrepareDepsOfTargetsUnderDirectoryKey) key.argument();
        PrepareDepsOfTargetsUnderDirectoryValue prepDepsValue =
            (PrepareDepsOfTargetsUnderDirectoryValue) subdirectorySkyValues.get(key);
        boolean packagesInSubdirectory = prepDepsValue.isDirectoryPackage();
        // If the subdirectory isn't a package, check to see if any of its subdirectories
        // transitively contain packages.
        if (!packagesInSubdirectory) {
          ImmutableCollection<Boolean> subdirectoryValues =
              prepDepsValue.getSubdirectoryTransitivelyContainsPackages().values();
          for (Boolean pkgsInSubSub : subdirectoryValues) {
            if (pkgsInSubSub) {
              packagesInSubdirectory = true;
              break;
            }
          }
        }
        builder.put(prepDepsKey.getRecursivePkgKey().getRootedPath(), packagesInSubdirectory);
      }
      return new PrepareDepsOfTargetsUnderDirectoryValue(visitor.isDirectoryPackage(),
          builder.build());
    }
  }

  private static class MyVisitor implements RecursiveDirectoryTraversalFunction.Visitor {

    private final FilteringPolicy filteringPolicy;
    private boolean isDirectoryPackage;

    private MyVisitor(FilteringPolicy filteringPolicy) {
      this.filteringPolicy = Preconditions.checkNotNull(filteringPolicy);
    }

    @Override
    public void visitPackageValue(Package pkg, Environment env) {
      isDirectoryPackage = true;
      loadTransitiveTargets(env, pkg, filteringPolicy);
    }

    public boolean isDirectoryPackage() {
      return isDirectoryPackage;
    }
  }

  private static void loadTransitiveTargets(Environment env, Package pkg,
      FilteringPolicy filteringPolicy) {
    ResolvedTargets<Target> packageTargets =
        TargetPatternResolverUtil.resolvePackageTargets(pkg, filteringPolicy);
    ImmutableList.Builder<SkyKey> builder = ImmutableList.builder();
    for (Target target : packageTargets.getTargets()) {
      builder.add(TransitiveTraversalValue.key(target.getLabel()));
    }
    ImmutableList<SkyKey> skyKeys = builder.build();
    env.getValuesOrThrow(skyKeys, NoSuchPackageException.class, NoSuchTargetException.class);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
