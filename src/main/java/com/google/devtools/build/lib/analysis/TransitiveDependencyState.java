// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static java.util.Comparator.comparing;

import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PrerequisitePackageFunction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Groups state associated with transitive dependencies. */
public final class TransitiveDependencyState {
  private final NestedSetBuilder<Cause> transitiveRootCauses;

  /**
   * State for constructing the packages transitively loaded for the value being built.
   *
   * <p>See {@link
   * com.google.devtools.build.lib.analysis.ConfiguredObjectValue#getTransitivePackages}.
   *
   * <p>Non-null when transitive packages are tracked, determined by {@link
   * com.google.devtools.build.lib.skyframe.SkyframeExecutor#shouldStoreTransitivePackagesInLoadingAndAnalysis}.
   */
  @Nullable private final PackageCollector packageCollector;

  /**
   * Retrieves packages that were previously requested by transitive dependencies.
   *
   * <p>When the {@link ConfiguredTargetFunction} computes a value, it depends on properties of its
   * dependencies. In some cases, those values are read directly out of the dependency's underlying
   * {@link Target}. All instances of this are to be restricted to where {@link
   * ConfiguredTargetAndData#target} is read.
   *
   * <p>More ideally, those properties would be conveyed via providers of those dependencies, but
   * doing so would adversely affect resting heap usage whereas {@link ConfiguredTargetAndData} is
   * ephemeral. Distributed implementations will include these properties in an extra provider. It
   * won't affect memory because the underlying package won't exist on the node loading it remotely.
   *
   * <p>It's valid to obtain {@link Package}s of dependencies from this function instead of creating
   * an edge in {@code Skyframe} due to the transitive dependency through the {@link
   * ConfiguredTarget}. Invalidation of the {@link Package} propagates upwards through the
   * dependency. This is compatible with bottom-up change pruning because {@link
   * ConfiguredTargetValue} uses identity equals.
   */
  private final PrerequisitePackageFunction prerequisitePackages;

  public TransitiveDependencyState(
      boolean storeTransitivePackages, PrerequisitePackageFunction prerequisitePackages) {
    this.transitiveRootCauses = NestedSetBuilder.stableOrder();
    this.packageCollector = storeTransitivePackages ? new PackageCollector() : null;
    this.prerequisitePackages = prerequisitePackages;
  }

  public static TransitiveDependencyState createForTesting() {
    return new TransitiveDependencyState(
        /* storeTransitivePackages= */ false,
        // Always returning null here causes the underlying code to fall back on declaring Package
        // edges for prerequisites, which is benign.
        /* prerequisitePackages= */ p -> null);
  }

  public NestedSetBuilder<Cause> transitiveRootCauses() {
    return transitiveRootCauses;
  }

  @Nullable
  public NestedSet<Package.Metadata> transitivePackages() {
    if (packageCollector == null) {
      return null;
    }
    return packageCollector.buildSet();
  }

  public void addTransitiveCauses(NestedSet<Cause> transitiveCauses) {
    transitiveRootCauses.addTransitive(transitiveCauses);
  }

  public void addTransitiveCause(Cause cause) {
    transitiveRootCauses.add(cause);
  }

  public boolean hasRootCause() {
    return !transitiveRootCauses.isEmpty();
  }

  public boolean storeTransitivePackages() {
    return packageCollector != null;
  }

  /** Adds to the set of transitive package metadata if {@link #storeTransitivePackages} is true. */
  public void updateTransitivePackages(Package.Metadata pkg) {
    if (packageCollector == null) {
      return;
    }
    packageCollector.packages.add(pkg);
  }

  /** Adds to the set of transitive package metadata if {@link #storeTransitivePackages} is true. */
  public void updateTransitivePackages(
      ConfiguredTargetKey key, NestedSet<Package.Metadata> packages) {
    if (packageCollector == null) {
      return;
    }
    packageCollector.configuredTargetPackages.put(key, packages);
  }

  /** Adds to the set of transitive package metadata if {@link #storeTransitivePackages} is true. */
  public void updateTransitivePackages(AspectKey key, NestedSet<Package.Metadata> packages) {
    if (packageCollector == null) {
      return;
    }
    packageCollector.aspectPackages.put(key, packages);
  }

  @Nullable
  public Package getDependencyPackage(PackageIdentifier packageId) throws InterruptedException {
    return prerequisitePackages.getExistingPackage(packageId);
  }

  /**
   * Collects package metadata of dependencies to be unified in a {@link NestedSet}.
   *
   * <p>Performs bookkeeping so the result is deterministic.
   *
   * <p>Work in Skyframe may complete in arbitrary order due to missing values and restarts. For
   * example, if a client requests {@code //foo} and {@code //bar}, it could receive any of the
   * following: {@code (//foo, null), (null, //bar), (//foo, //bar) or (null, null)}.
   *
   * <p>This class tracks how the {@link Package}s are added so they can be given a deterministic
   * order. This is required for determinism of {@link ActionKeyComputer#computeKey}.
   */
  private static class PackageCollector {
    /**
     * Keeps packages that were added directly as a list.
     *
     * <p>These will be sorted.
     */
    private final ArrayList<Package.Metadata> packages = new ArrayList<>();

    /** Stores transitive {@link Package.Metadata}s of {@link ConfiguredTargetValues}s. */
    private final TreeMap<ConfiguredTargetKey, NestedSet<Package.Metadata>>
        configuredTargetPackages = new TreeMap<>(ConfiguredTargetKey.ORDERING);

    /** Stores transitive {@link Package.Metadata}s of {@link AspectValue}s. */
    private final TreeMap<AspectKey, NestedSet<Package.Metadata>> aspectPackages =
        new TreeMap<>(AspectKey.ORDERING);

    /**
     * Constructs the deterministically ordered result.
     *
     * <p>It's safe to call this multiple times.
     */
    private NestedSet<Package.Metadata> buildSet() {
      var result = NestedSetBuilder.<Package.Metadata>stableOrder();

      Collections.sort(packages, comparing(Package.Metadata::packageIdentifier));
      result.addAll(packages);

      for (NestedSet<Package.Metadata> packageSet : configuredTargetPackages.values()) {
        result.addTransitive(packageSet);
      }
      for (NestedSet<Package.Metadata> packageSet : aspectPackages.values()) {
        result.addTransitive(packageSet);
      }

      return result.build();
    }
  }

}
