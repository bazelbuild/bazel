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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.protobuf.Duration;
import java.util.Collection;
import java.util.Map;

/** Interface encapsulating the strategy used for recording Package Metrics. */
public interface PackageMetricsRecorder {

  /** What type of packages are metrics being recorded for? */
  enum Type {
    ONLY_EXTREMES,
    ALL,
  }

  /** Records the metrics for a given package. */
  void recordMetrics(PackageIdentifier pkgId, PackageLoadMetrics metrics);

  /**
   * Returns a {@code Map<PackageIdentifier, Duration>} of recorded load durations. This may contain
   * only a subset of all packages loaded based on the implementation.
   */
  Map<PackageIdentifier, Duration> getLoadTimes();

  /**
   * Returns a {@code Map<PackageIdentifier, Long>} of glob costs. This may contain only a subset of
   * all packages loaded based on the implementation.
   */
  Map<PackageIdentifier, Long> getGlobFilesystemOperationCost();

  /**
   * Returns a {@code Map<PackageIdentifier, Long>} of computation steps. This may contain only a
   * subset of all packages loaded based on the implementation.
   */
  Map<PackageIdentifier, Long> getComputationSteps();

  /**
   * Returns a {@code Map<PackageIdentifier, Long>} of num targets. This may contain only a subset
   * of all packages loaded based on the implementation.
   */
  Map<PackageIdentifier, Long> getNumTargets();

  /**
   * Returns a {@code Map<PackageIdentifier, Long>} of num targets. This may contain only a subset
   * of all packages loaded based on the implementation.
   */
  Map<PackageIdentifier, Long> getNumTransitiveLoads();

  /** Returns map of package overhead. This may contain only a subset of all packages loaded. */
  Map<PackageIdentifier, Long> getPackageOverhead();

  /** Clears the contents of the PackageMetricsRecorder. */
  void clear();

  /**
   * Called after package loading is complete to allow handlers to perform post-loading phase
   * processing.
   */
  void loadingFinished();

  /** Returns the type of package metrics being recorded. */
  Type getRecorderType();

  /** If Type is ALL returns metrics for all Packages loaded. */
  Collection<PackageLoadMetrics> getPackageLoadMetrics();

  /* TODO(twerth): Remove method after migration is complete. */
  default ImmutableCollection<PackageMetrics> getPackageMetrics() {
    Collection<PackageLoadMetrics> packageLoadMetrics = getPackageLoadMetrics();
    return packageLoadMetrics.stream()
        .map(
            plm ->
                PackageMetrics.newBuilder()
                    .setName(plm.getName())
                    .setPackageOverhead(plm.getPackageOverhead())
                    .setComputationSteps(plm.getComputationSteps())
                    .setLoadDuration(plm.getLoadDuration())
                    .setNumTargets(plm.getNumTargets())
                    .setNumTransitiveLoads(plm.getNumTransitiveLoads())
                    .setGlobFilesystemOperationCost(plm.getGlobFilesystemOperationCost())
                    .build())
        .collect(toImmutableList());
  }
}
