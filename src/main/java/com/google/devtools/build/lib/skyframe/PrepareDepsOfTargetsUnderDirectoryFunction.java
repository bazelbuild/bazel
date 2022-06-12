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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.skyframe.ProcessPackageDirectory.ProcessPackageDirectorySkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeIterableResult;

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
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, ProcessPackageDirectorySkyFunctionException {
    PrepareDepsOfTargetsUnderDirectoryKey argument =
        (PrepareDepsOfTargetsUnderDirectoryKey) skyKey.argument();
    final FilteringPolicy filteringPolicy = argument.getFilteringPolicy();
    RecursivePkgKey recursivePkgKey = argument.getRecursivePkgKey();
    ProcessPackageDirectory processPackageDirectory =
        new ProcessPackageDirectory(
            directories,
            (repository, subdirectory, excludedSubdirectoriesBeneathSubdirectory) ->
                PrepareDepsOfTargetsUnderDirectoryValue.key(
                    repository,
                    subdirectory,
                    excludedSubdirectoriesBeneathSubdirectory,
                    filteringPolicy));
    ProcessPackageDirectoryResult packageExistenceAndSubdirDeps =
        processPackageDirectory.getPackageExistenceAndSubdirDeps(
            recursivePkgKey.getRootedPath(),
            recursivePkgKey.getRepositoryName(),
            recursivePkgKey.getExcludedPaths(),
            env);
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
                          recursivePkgKey.getRepositoryName(),
                          recursivePkgKey.getRootedPath().getRootRelativePath()),
                      filteringPolicy)),
              keysToRequest);
    }
    SkyframeIterableResult result = env.getOrderedValuesAndExceptions(keysToRequest);
    if (env.valuesMissing()) {
      return null;
    }
    if (checkValuesMissing(keysToRequest, result)) {
      return null;
    }
    return PrepareDepsOfTargetsUnderDirectoryValue.INSTANCE;
  }

  /**
   * Returns true iff {@link SkyframeIterableResult#nextOrThrow} returned {@code null} that
   * indicated a missing dependency.
   */
  static boolean checkValuesMissing(
      Iterable<? extends SkyKey> keysToRequest, SkyframeIterableResult result) {
    while (result.hasNext()) {
      try {
        SkyValue value = result.nextOrThrow(NoSuchPackageException.class);
        if (value == null) {
          BugReport.sendBugReport(
              new IllegalStateException(
                  "Some value from "
                      + Iterables.toString(keysToRequest)
                      + " was missing, this should never happen"));
          return true;
        }
      } catch (NoSuchPackageException e) {
        continue;
      }
    }
    return false;
  }
}
