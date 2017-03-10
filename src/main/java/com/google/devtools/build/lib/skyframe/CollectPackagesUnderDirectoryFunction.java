// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
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
 * Computes {@link CollectPackagesUnderDirectoryValue} which describes whether the directory is a
 * package, or would have been a package but for a package loading error, and whether non-excluded
 * packages (or errors) exist below each of the directory's subdirectories. As a side effect, loads
 * all of these packages, in order to interleave the disk-bound work of checking for directories and
 * the CPU-bound work of package loading.
 */
public class CollectPackagesUnderDirectoryFunction implements SkyFunction {
  private final BlazeDirectories directories;

  public CollectPackagesUnderDirectoryFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    return new MyTraversalFunction().visitDirectory((RecursivePkgKey) skyKey.argument(), env);
  }

  private class MyTraversalFunction
      extends RecursiveDirectoryTraversalFunction<MyVisitor, CollectPackagesUnderDirectoryValue> {

    private MyTraversalFunction() {
      super(directories);
    }

    @Override
    protected MyVisitor getInitialVisitor() {
      return new MyVisitor();
    }

    @Override
    protected SkyKey getSkyKeyForSubdirectory(
        RepositoryName repository,
        RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory) {
      return CollectPackagesUnderDirectoryValue.key(
          repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory);
    }

    @Override
    protected CollectPackagesUnderDirectoryValue aggregateWithSubdirectorySkyValues(
        MyVisitor visitor, Map<SkyKey, SkyValue> subdirectorySkyValues) {
      // Aggregate the child subdirectory package state.
      ImmutableMap.Builder<RootedPath, Boolean> builder = ImmutableMap.builder();
      for (SkyKey key : subdirectorySkyValues.keySet()) {
        RecursivePkgKey recursivePkgKey = (RecursivePkgKey) key.argument();
        CollectPackagesUnderDirectoryValue collectPackagesValue =
            (CollectPackagesUnderDirectoryValue) subdirectorySkyValues.get(key);

        boolean packagesOrErrorsInSubdirectory =
            collectPackagesValue.isDirectoryPackage()
                || collectPackagesValue.getErrorMessage() != null
                || Iterables.contains(
                    collectPackagesValue
                        .getSubdirectoryTransitivelyContainsPackagesOrErrors()
                        .values(),
                    Boolean.TRUE);

        builder.put(recursivePkgKey.getRootedPath(), packagesOrErrorsInSubdirectory);
      }
      ImmutableMap<RootedPath, Boolean> subdirectories = builder.build();
      String errorMessage = visitor.getErrorMessage();
      if (errorMessage != null) {
        return CollectPackagesUnderDirectoryValue.ofError(errorMessage, subdirectories);
      }
      return CollectPackagesUnderDirectoryValue.ofNoError(
          visitor.isDirectoryPackage(), subdirectories);
    }
  }

  private static class MyVisitor implements RecursiveDirectoryTraversalFunction.Visitor {

    private boolean isDirectoryPackage;
    @Nullable private String errorMessage;

    private MyVisitor() {}

    @Override
    public void visitPackageValue(Package pkg, Environment env) {
      isDirectoryPackage = true;
    }

    @Override
    public void visitPackageError(NoSuchPackageException e, Environment env)
        throws InterruptedException {
      errorMessage = e.getMessage();
    }

    boolean isDirectoryPackage() {
      return isDirectoryPackage;
    }

    @Nullable
    String getErrorMessage() {
      return errorMessage;
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
