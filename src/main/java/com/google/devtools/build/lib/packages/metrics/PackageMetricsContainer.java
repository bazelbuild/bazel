// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.metrics;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.protobuf.util.Durations;
import java.util.Comparator;

/** Container class holding a PackageIdentifier and PackageMetrics proto. */
@AutoValue
public abstract class PackageMetricsContainer {

  /** Sorts by LoadTime Duration. */
  public static final Comparator<PackageMetricsContainer> LOAD_TIMES_COMP =
      Comparator.comparing(
          c -> c.getPackageMetricsInternal().getLoadDuration(), Durations.comparator());
  /** Sorts by Num Target count . */
  public static final Comparator<PackageMetricsContainer> NUM_TARGETS_COMP =
      Comparator.comparingLong(c -> c.getPackageMetricsInternal().getNumTargets());
  /** Sorts by Comutation Steps count. */
  public static final Comparator<PackageMetricsContainer> COMPUTATION_STEPS_COMP =
      Comparator.comparingLong(c -> c.getPackageMetricsInternal().getComputationSteps());
  /** Sorts by Transitive Load Count. */
  public static final Comparator<PackageMetricsContainer> TRANSITIVE_LOADS_COMP =
      Comparator.comparingLong(c -> c.getPackageMetricsInternal().getNumTransitiveLoads());
  /** Sorts by Package Overhead. */
  public static final Comparator<PackageMetricsContainer> OVERHEAD_COMP =
      Comparator.comparingLong(c -> c.getPackageMetricsInternal().getPackageOverhead());

  public static PackageMetricsContainer create(PackageIdentifier pkgId, PackageMetrics metrics) {
    return new AutoValue_PackageMetricsContainer(pkgId, metrics);
  }

  public abstract PackageIdentifier getPackageIdentifier();

  abstract PackageMetrics getPackageMetricsInternal();

  /** Construct a full PackageMetrics object with the name set lazily from the PackageIdentifier. */
  public PackageMetrics getPackageMetrics() {
    return getPackageMetricsInternal().toBuilder()
        .setName(getPackageIdentifier().toString())
        .build();
  }
}
