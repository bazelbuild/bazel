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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * RecursivePkgFunction builds up the set of packages underneath a given directory
 * transitively.
 *
 * <p>Example: foo/BUILD, foo/sub/x, foo/subpkg/BUILD would yield transitive packages "foo" and
 * "foo/subpkg".
 */
public class RecursivePkgFunction implements SkyFunction {
  private final BlazeDirectories directories;

  public RecursivePkgFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    return new MyTraversalFunction().visitDirectory((RecursivePkgKey) skyKey.argument(), env);
  }

  private class MyTraversalFunction
      extends RecursiveDirectoryTraversalFunction<MyVisitor, RecursivePkgValue> {

    private MyTraversalFunction() {
      super(directories);
    }

    @Override
    protected RecursivePkgValue getEmptyReturn() {
      return RecursivePkgValue.EMPTY;
    }

    @Override
    protected MyVisitor getInitialVisitor() {
      return new MyVisitor();
    }

    @Override
    protected SkyKey getSkyKeyForSubdirectory(RepositoryName repository, RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
      return RecursivePkgValue.key(
          repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory);
    }

    @Override
    protected RecursivePkgValue aggregateWithSubdirectorySkyValues(MyVisitor visitor,
        Map<SkyKey, SkyValue> subdirectorySkyValues) {
      // Aggregate the transitive subpackages.
      for (SkyValue childValue : subdirectorySkyValues.values()) {
        if (childValue != null) {
          visitor.addTransitivePackages(((RecursivePkgValue) childValue).getPackages());
        }
      }
      return visitor.createRecursivePkgValue();
    }
  }

  private static class MyVisitor implements RecursiveDirectoryTraversalFunction.Visitor {

    private final NestedSetBuilder<String> packages = new NestedSetBuilder<>(Order.STABLE_ORDER);

    @Override
    public void visitPackageValue(Package pkg, Environment env) {
      packages.add(pkg.getName());
    }

    void addTransitivePackages(NestedSet<String> transitivePackages) {
      packages.addTransitive(transitivePackages);
    }

    RecursivePkgValue createRecursivePkgValue() {
      return RecursivePkgValue.create(packages);
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
