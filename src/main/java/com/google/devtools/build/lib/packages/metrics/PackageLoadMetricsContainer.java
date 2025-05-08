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
public abstract class PackageLoadMetricsContainer {

  /** Sorts by LoadTime Duration. */
  public static final Comparator<PackageLoadMetricsContainer> LOAD_TIMES_COMP =
      Comparator.comparing(
          c -> c.getPackageLoadMetricsInternal().getLoadDuration(), Durations.comparator());
  /** Sorts by Num Target count . */
  public static final Comparator<PackageLoadMetricsContainer> NUM_TARGETS_COMP =
      Comparator.comparingLong(c -> c.getPackageLoadMetricsInternal().getNumTargets());
  /** Sorts by Comutation Steps count. */
  public static final Comparator<PackageLoadMetricsContainer> COMPUTATION_STEPS_COMP =
      Comparator.comparingLong(c -> c.getPackageLoadMetricsInternal().getComputationSteps());
  /** Sorts by Transitive Load Count. */
  public static final Comparator<PackageLoadMetricsContainer> TRANSITIVE_LOADS_COMP =
      Comparator.comparingLong(c -> c.getPackageLoadMetricsInternal().getNumTransitiveLoads());
  /** Sorts by Package Overhead. */
  public static final Comparator<PackageLoadMetricsContainer> OVERHEAD_COMP =
      Comparator.comparingLong(c -> c.getPackageLoadMetricsInternal().getPackageOverhead());

  public static PackageLoadMetricsContainer create(
      PackageIdentifier pkgId, PackageLoadMetrics metrics) {
    return new AutoValue_PackageLoadMetricsContainer(pkgId, metrics);
  }

  public abstract PackageIdentifier getPackageIdentifier();

  abstract PackageLoadMetrics getPackageLoadMetricsInternal();

  /** Construct a full PackageMetrics object with the name set lazily from the PackageIdentifier. */
  public PackageLoadMetrics getPackageLoadMetrics() {
    return getPackageLoadMetricsInternal().toBuilder()
        .setName(getPackageIdentifier().toString())
        .build();
  }
}
