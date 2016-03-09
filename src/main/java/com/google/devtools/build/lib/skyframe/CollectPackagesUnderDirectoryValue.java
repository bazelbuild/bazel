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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

/**
 * The value computed by {@link CollectPackagesUnderDirectoryFunction}. Contains a mapping for all
 * its non-excluded directories to whether there are packages beneath them.
 *
 * <p>This value is used by {@link GraphBackedRecursivePackageProvider#getPackagesUnderDirectory}
 * to help it traverse the graph and find the set of packages under a directory, recursively by
 * {@link CollectPackagesUnderDirectoryFunction} which computes a value for a directory by
 * aggregating results calculated from its subdirectories, and by
 * {@link PrepareDepsOfTargetsUnderDirectoryFunction} which uses this value to find transitive
 * targets to load.
 *
 * <p>Note that even though the {@link CollectPackagesUnderDirectoryFunction} is evaluated in
 * part because of its side-effects (i.e. loading transitive dependencies of targets), this value
 * interacts safely with change pruning, despite the fact that this value is a lossy representation
 * of the packages beneath a directory (i.e. it doesn't care <b>which</b> packages are under a
 * directory, just whether there are any). When the targets in a package change, the
 * {@link PackageValue} that {@link CollectPackagesUnderDirectoryFunction} depends on will be
 * invalidated, and the PrepareDeps function for that package's directory will be reevaluated,
 * loading any new transitive dependencies. Change pruning may prevent the reevaluation of
 * PrepareDeps for directories above that one, but they don't need to be re-run.
 */
public class CollectPackagesUnderDirectoryValue implements SkyValue {
  public static final CollectPackagesUnderDirectoryValue EMPTY =
      new CollectPackagesUnderDirectoryValue(false, ImmutableMap.<RootedPath, Boolean>of());

  private final boolean isDirectoryPackage;
  private final ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages;

  private CollectPackagesUnderDirectoryValue(
      boolean isDirectoryPackage,
      ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages) {
    this.subdirectoryTransitivelyContainsPackages = subdirectoryTransitivelyContainsPackages;
    this.isDirectoryPackage = isDirectoryPackage;
  }

  public static CollectPackagesUnderDirectoryValue of(
      boolean isDirectoryPackage,
      ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackages) {
    if (!isDirectoryPackage && subdirectoryTransitivelyContainsPackages.isEmpty()) {
      return EMPTY;
    }
    return new CollectPackagesUnderDirectoryValue(
        isDirectoryPackage, subdirectoryTransitivelyContainsPackages);
  }

  public boolean isDirectoryPackage() {
    return isDirectoryPackage;
  }

  public ImmutableMap<RootedPath, Boolean> getSubdirectoryTransitivelyContainsPackages() {
    return subdirectoryTransitivelyContainsPackages;
  }

  @Override
  public int hashCode() {
    return Objects.hash(isDirectoryPackage, subdirectoryTransitivelyContainsPackages);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof CollectPackagesUnderDirectoryValue)) {
      return false;
    }
    CollectPackagesUnderDirectoryValue that = (CollectPackagesUnderDirectoryValue) o;
    return this.isDirectoryPackage == that.isDirectoryPackage
        && this
            .subdirectoryTransitivelyContainsPackages.equals(
                that.subdirectoryTransitivelyContainsPackages);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("isDirectoryPackage", isDirectoryPackage)
        .add("subdirectoryTransitivelyContainsPackages", subdirectoryTransitivelyContainsPackages)
        .toString();
  }

  /** Create a collect packages under directory request. */
  @ThreadSafe
  static SkyKey key(
      RepositoryName repository, RootedPath rootedPath, ImmutableSet<PathFragment> excludedPaths) {
    return key(new RecursivePkgKey(repository, rootedPath, excludedPaths));
  }

  static SkyKey key(RecursivePkgKey recursivePkgKey) {
    return SkyKey.create(SkyFunctions.COLLECT_PACKAGES_UNDER_DIRECTORY, recursivePkgKey);
  }
}
