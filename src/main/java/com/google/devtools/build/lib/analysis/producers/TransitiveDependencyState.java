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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Package;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** Tuple storing state associated with transitive dependencies. */
public final class TransitiveDependencyState {
  private final NestedSetBuilder<Cause> transitiveRootCauses;

  /**
   * The set of packages transitively loaded for the value being built.
   *
   * <p>See {@link
   * com.google.devtools.build.lib.analysis.ConfiguredObjectValue#getTransitivePackages}.
   *
   * <p>Non-null when transitive packages are tracked, determined by {@link
   * com.google.devtools.build.lib.skyframe.SkyframeExecutor#shouldStoreTransitivePackagesInLoadingAndAnalysis}.
   */
  @Nullable private final NestedSetBuilder<Package> transitivePackages;

  /**
   * Contains packages that were previously computed.
   *
   * <p>It's valid to obtain packages of dependencies from this map instead of creating an edge in
   * {@code Skyframe} due to the transitive dependency through the {@link ConfiguredTarget}. Note
   * that this is compatible with bottom-up change pruning because {@link ConfiguredTargetValue}
   * uses identity equals.
   */
  @Nullable // TODO(b/261521010): make this non-null.
  private final ConcurrentHashMap<PackageIdentifier, Package> prerequisitePackages;

  public TransitiveDependencyState(
      NestedSetBuilder<Cause> transitiveRootCauses,
      @Nullable NestedSetBuilder<Package> transitivePackages,
      @Nullable ConcurrentHashMap<PackageIdentifier, Package> prerequisitePackages) {
    this.transitiveRootCauses = transitiveRootCauses;
    this.transitivePackages = transitivePackages;
    this.prerequisitePackages = prerequisitePackages;
  }

  public NestedSetBuilder<Package> transitivePackages() {
    return transitivePackages;
  }

  public void addTransitiveCauses(NestedSet<Cause> transitiveCauses) {
    transitiveRootCauses.addTransitive(transitiveCauses);
  }

  /**
   * Adds to the set of transitive packages if tracked.
   *
   * <p>This is a no-op otherwise.
   */
  public void updateTransitivePackages(ConfiguredTargetValue configuredTarget) {
    if (transitivePackages == null) {
      return;
    }
    transitivePackages.addTransitive(configuredTarget.getTransitivePackages());
  }

  @Nullable
  public Package getDependencyPackage(PackageIdentifier packageId) {
    if (prerequisitePackages == null) {
      return null;
    }
    return prerequisitePackages.get(packageId);
  }
}
