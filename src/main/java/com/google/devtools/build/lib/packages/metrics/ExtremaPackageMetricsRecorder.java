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
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Streams;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.protobuf.Duration;
import com.google.protobuf.util.Durations;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.concurrent.GuardedBy;

/** Tracks per-invocation extreme package loading events. */
public class ExtremaPackageMetricsRecorder implements PackageMetricsRecorder {
  private final int currentNumPackagesToTrack;
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @GuardedBy("this")
  private final Extrema<PackageLoadMetricsContainer> slowestPackagesToLoad;

  @GuardedBy("this")
  private final Extrema<PackageLoadMetricsContainer> largestPackages;

  @GuardedBy("this")
  private final Extrema<PackageLoadMetricsContainer> packagesWithMostTransitiveLoads;

  @GuardedBy("this")
  private final Extrema<PackageLoadMetricsContainer> packagesWithMostComputationSteps;

  @GuardedBy("this")
  private final Extrema<PackageLoadMetricsContainer> packagesWithMostOverhead;

  ExtremaPackageMetricsRecorder(int currentNumPackagesToTrack) {
    Preconditions.checkArgument(currentNumPackagesToTrack >= 0, "num packages must be >= 0");
    this.currentNumPackagesToTrack = currentNumPackagesToTrack;
    this.slowestPackagesToLoad =
        Extrema.max(currentNumPackagesToTrack, PackageLoadMetricsContainer.LOAD_TIMES_COMP);
    this.largestPackages =
        Extrema.max(currentNumPackagesToTrack, PackageLoadMetricsContainer.NUM_TARGETS_COMP);
    this.packagesWithMostTransitiveLoads =
        Extrema.max(currentNumPackagesToTrack, PackageLoadMetricsContainer.TRANSITIVE_LOADS_COMP);
    this.packagesWithMostComputationSteps =
        Extrema.max(currentNumPackagesToTrack, PackageLoadMetricsContainer.COMPUTATION_STEPS_COMP);
    this.packagesWithMostOverhead =
        Extrema.max(currentNumPackagesToTrack, PackageLoadMetricsContainer.OVERHEAD_COMP);
  }

  public int getNumPackagesToTrack() {
    return currentNumPackagesToTrack;
  }

  @Override
  public synchronized void recordMetrics(PackageIdentifier pkgId, PackageLoadMetrics metrics) {
    PackageLoadMetricsContainer cont = PackageLoadMetricsContainer.create(pkgId, metrics);
    slowestPackagesToLoad.aggregate(cont);
    packagesWithMostComputationSteps.aggregate(cont);
    largestPackages.aggregate(cont);
    packagesWithMostTransitiveLoads.aggregate(cont);
    if (metrics.hasPackageOverhead()) {
      packagesWithMostOverhead.aggregate(cont);
    }
  }

  @Override
  public synchronized Map<PackageIdentifier, Duration> getLoadTimes() {
    return slowestPackagesToLoad.getExtremeElements().stream()
        .collect(
            Collectors.toMap(
                PackageLoadMetricsContainer::getPackageIdentifier,
                v -> v.getPackageLoadMetricsInternal().getLoadDuration(),
                (k, v) -> v,
                LinkedHashMap::new)); // use a LinkedHashMap to ensure iteration order is maintained
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getComputationSteps() {
    return toMap(packagesWithMostComputationSteps, PackageLoadMetrics::getComputationSteps);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTargets() {
    return toMap(largestPackages, PackageLoadMetrics::getNumTargets);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTransitiveLoads() {
    return toMap(packagesWithMostTransitiveLoads, PackageLoadMetrics::getNumTransitiveLoads);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getPackageOverhead() {
    return toMap(packagesWithMostOverhead, PackageLoadMetrics::getPackageOverhead);
  }

  private synchronized Map<PackageIdentifier, Long> toMap(
      Extrema<PackageLoadMetricsContainer> ext, Function<PackageLoadMetrics, Long> fn) {

    return ext.getExtremeElements().stream()
        .collect(
            Collectors.toMap(
                PackageLoadMetricsContainer::getPackageIdentifier,
                v -> fn.apply(v.getPackageLoadMetricsInternal()),
                (k, v) -> v,
                LinkedHashMap::new)); // use a LinkedHashMap to ensure iteration order is maintained
  }

  @Override
  public synchronized void clear() {
    slowestPackagesToLoad.clear();
    packagesWithMostComputationSteps.clear();
    largestPackages.clear();
    packagesWithMostTransitiveLoads.clear();
    packagesWithMostOverhead.clear();
  }

  @Override
  public synchronized void loadingFinished() {
    logIfNonEmpty(
        "Slowest packages (ms)",
        slowestPackagesToLoad.getExtremeElements(),
        c -> Durations.toMillis(c.getPackageLoadMetricsInternal().getLoadDuration()));
    logIfNonEmpty(
        "Largest packages (num targets)",
        largestPackages.getExtremeElements(),
        c -> c.getPackageLoadMetricsInternal().getNumTargets());
    logIfNonEmpty(
        "Packages with most computation steps",
        packagesWithMostComputationSteps.getExtremeElements(),
        c -> c.getPackageLoadMetricsInternal().getComputationSteps());
    logIfNonEmpty(
        "Packages with most transitive loads (num bzl files)",
        packagesWithMostTransitiveLoads.getExtremeElements(),
        c -> c.getPackageLoadMetricsInternal().getNumTransitiveLoads());
    logIfNonEmpty(
        "Packages with most overhead",
        packagesWithMostOverhead.getExtremeElements(),
        c -> c.getPackageLoadMetricsInternal().getPackageOverhead());
    clear();
  }

  @Override
  public Type getRecorderType() {
    return PackageMetricsRecorder.Type.ONLY_EXTREMES;
  }

  @Override
  public synchronized Collection<PackageLoadMetrics> getPackageLoadMetrics() {
    return Streams.concat(
            slowestPackagesToLoad.getExtremeElements().stream(),
            packagesWithMostComputationSteps.getExtremeElements().stream(),
            largestPackages.getExtremeElements().stream(),
            packagesWithMostTransitiveLoads.getExtremeElements().stream(),
            packagesWithMostOverhead.getExtremeElements().stream())
        .map(PackageLoadMetricsContainer::getPackageLoadMetrics)
        .collect(toImmutableSet());
  }

  private static void logIfNonEmpty(
      String logLinePrefix,
      List<PackageLoadMetricsContainer> extremeElements,
      Function<PackageLoadMetricsContainer, Long> valueMapper) {
    List<String> logString =
        extremeElements.stream()
            .map(v -> String.format("%s (%d)", v.getPackageIdentifier(), valueMapper.apply(v)))
            .collect(toImmutableList());
    if (!extremeElements.isEmpty()) {
      logger.atInfo().log("%s: %s", logLinePrefix, Joiner.on(", ").join(logString));
    }
  }
}
