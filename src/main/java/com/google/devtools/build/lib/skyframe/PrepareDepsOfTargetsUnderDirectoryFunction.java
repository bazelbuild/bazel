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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.util.Preconditions;
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
 */
public class PrepareDepsOfTargetsUnderDirectoryFunction implements SkyFunction {
  private final BlazeDirectories directories;

  public PrepareDepsOfTargetsUnderDirectoryFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    PrepareDepsOfTargetsUnderDirectoryKey argument =
        (PrepareDepsOfTargetsUnderDirectoryKey) skyKey.argument();
    FilteringPolicy filteringPolicy = argument.getFilteringPolicy();
    RecursivePkgKey recursivePkgKey = argument.getRecursivePkgKey();
    return new MyTraversalFunction(filteringPolicy).visitDirectory(recursivePkgKey, env);
  }

  private class MyTraversalFunction
      extends RecursiveDirectoryTraversalFunction<
          MyVisitor, PrepareDepsOfTargetsUnderDirectoryValue> {
    private final FilteringPolicy filteringPolicy;

    private MyTraversalFunction(FilteringPolicy filteringPolicy) {
      super(directories);
      this.filteringPolicy = filteringPolicy;
    }

    @Override
    protected PrepareDepsOfTargetsUnderDirectoryValue getEmptyReturn() {
      return PrepareDepsOfTargetsUnderDirectoryValue.INSTANCE;
    }

    @Override
    protected MyVisitor getInitialVisitor() {
      return new MyVisitor(filteringPolicy);
    }

    @Override
    protected SkyKey getSkyKeyForSubdirectory(
        RepositoryName repository,
        RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
      return PrepareDepsOfTargetsUnderDirectoryValue.key(
          repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory, filteringPolicy);
    }

    @Override
    protected PrepareDepsOfTargetsUnderDirectoryValue aggregateWithSubdirectorySkyValues(
        MyVisitor visitor, Map<SkyKey, SkyValue> subdirectorySkyValues) {
      return PrepareDepsOfTargetsUnderDirectoryValue.INSTANCE;
    }
  }

  private static class MyVisitor implements RecursiveDirectoryTraversalFunction.Visitor {
    private final FilteringPolicy filteringPolicy;

    private MyVisitor(FilteringPolicy filteringPolicy) {
      this.filteringPolicy = Preconditions.checkNotNull(filteringPolicy);
    }

    @Override
    public void visitPackageValue(Package pkg, Environment env) {
      loadTransitiveTargets(env, pkg, filteringPolicy);
    }
  }

  private static void loadTransitiveTargets(
      Environment env, Package pkg, FilteringPolicy filteringPolicy) {
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
