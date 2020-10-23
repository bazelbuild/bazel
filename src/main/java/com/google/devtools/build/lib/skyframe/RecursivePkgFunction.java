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

  /**
   * N.B.: May silently throw {@link com.google.devtools.build.lib.packages.NoSuchPackageException}
   * in nokeep_going mode!
   */
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    return new MyTraversalFunction().visitDirectory((RecursivePkgKey) skyKey.argument(), env);
  }

  private class MyTraversalFunction
      extends RecursiveDirectoryTraversalFunction<MyPackageDirectoryConsumer, RecursivePkgValue> {

    private MyTraversalFunction() {
      super(directories);
    }

    @Override
    protected MyPackageDirectoryConsumer getInitialConsumer() {
      return new MyPackageDirectoryConsumer();
    }

    @Override
    protected SkyKey getSkyKeyForSubdirectory(RepositoryName repository, RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
      return RecursivePkgValue.key(
          repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory);
    }

    @Override
    protected RecursivePkgValue aggregateWithSubdirectorySkyValues(
        MyPackageDirectoryConsumer consumer, Map<SkyKey, SkyValue> subdirectorySkyValues) {
      // Aggregate the transitive subpackages.
      for (SkyValue childValue : subdirectorySkyValues.values()) {
        consumer.addTransitivePackages(((RecursivePkgValue) childValue).getPackages());
        if (((RecursivePkgValue) childValue).hasErrors()) {
          consumer.addTransitiveErrors();
        }
      }
      return consumer.createRecursivePkgValue();
    }
  }

  private static class MyPackageDirectoryConsumer
      implements RecursiveDirectoryTraversalFunction.PackageDirectoryConsumer {

    private final NestedSetBuilder<String> packages = new NestedSetBuilder<>(Order.STABLE_ORDER);
    private boolean hasErrors = false;

    @Override
    public void notePackage(PathFragment pkgPath) {
      packages.add(pkgPath.getPathString());
    }

    @Override
    public void notePackageError(String noSuchPackageExceptionErrorMessage) {
      hasErrors = true;
    }

    void addTransitivePackages(NestedSet<String> transitivePackages) {
      packages.addTransitive(transitivePackages);
    }

    void addTransitiveErrors() {
      hasErrors = true;
    }

    RecursivePkgValue createRecursivePkgValue() {
      return RecursivePkgValue.create(packages, hasErrors);
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
