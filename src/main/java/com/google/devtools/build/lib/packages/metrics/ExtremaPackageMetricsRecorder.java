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
class ExtremaPackageMetricsRecorder implements PackageMetricsRecorder {
  private final int currentNumPackagesToTrack;
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @GuardedBy("this")
  private final Extrema<PackageMetricsContainer> slowestPackagesToLoad;

  @GuardedBy("this")
  private final Extrema<PackageMetricsContainer> largestPackages;

  @GuardedBy("this")
  private final Extrema<PackageMetricsContainer> packagesWithMostTransitiveLoads;

  @GuardedBy("this")
  private final Extrema<PackageMetricsContainer> packagesWithMostComputationSteps;

  ExtremaPackageMetricsRecorder(int currentNumPackagesToTrack) {
    Preconditions.checkArgument(currentNumPackagesToTrack >= 0, "num packages must be >= 0");
    this.currentNumPackagesToTrack = currentNumPackagesToTrack;
    this.slowestPackagesToLoad =
        Extrema.max(currentNumPackagesToTrack, PackageMetricsContainer.LOAD_TIMES_COMP);
    this.largestPackages =
        Extrema.max(currentNumPackagesToTrack, PackageMetricsContainer.NUM_TARGETS_COMP);
    this.packagesWithMostTransitiveLoads =
        Extrema.max(currentNumPackagesToTrack, PackageMetricsContainer.TRANSITIVE_LOADS_COMP);
    this.packagesWithMostComputationSteps =
        Extrema.max(currentNumPackagesToTrack, PackageMetricsContainer.COMPUTATION_STEPS_COMP);
  }

  public int getNumPackageToTrack() {
    return currentNumPackagesToTrack;
  }

  @Override
  public synchronized void recordMetrics(PackageIdentifier pkgId, PackageMetrics metrics) {
    PackageMetricsContainer cont = PackageMetricsContainer.create(pkgId, metrics);
    slowestPackagesToLoad.aggregate(cont);
    packagesWithMostComputationSteps.aggregate(cont);
    largestPackages.aggregate(cont);
    packagesWithMostTransitiveLoads.aggregate(cont);
  }

  @Override
  public synchronized Map<PackageIdentifier, Duration> getLoadTimes() {
    return slowestPackagesToLoad.getExtremeElements().stream()
        .collect(
            Collectors.toMap(
                PackageMetricsContainer::getPackageIdentifier,
                v -> v.getPackageMetricsInternal().getLoadDuration(),
                (k, v) -> v,
                LinkedHashMap::new)); // use a LinkedHashMap to ensure iteration order is maintained
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getComputationSteps() {
    return toMap(packagesWithMostComputationSteps, PackageMetrics::getComputationSteps);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTargets() {
    return toMap(largestPackages, PackageMetrics::getNumTargets);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTransitiveLoads() {
    return toMap(packagesWithMostTransitiveLoads, PackageMetrics::getNumTransitiveLoads);
  }

  private synchronized Map<PackageIdentifier, Long> toMap(
      Extrema<PackageMetricsContainer> ext, Function<PackageMetrics, Long> fn) {

    return ext.getExtremeElements().stream()
        .collect(
            Collectors.toMap(
                PackageMetricsContainer::getPackageIdentifier,
                v -> fn.apply(v.getPackageMetricsInternal()),
                (k, v) -> v,
                LinkedHashMap::new)); // use a LinkedHashMap to ensure iteration order is maintained
  }

  @Override
  public synchronized void clear() {
    slowestPackagesToLoad.clear();
    packagesWithMostComputationSteps.clear();
    largestPackages.clear();
    packagesWithMostTransitiveLoads.clear();
  }

  @Override
  public synchronized void loadingFinished() {
    logIfNonEmpty(
        "Slowest packages (ms)",
        slowestPackagesToLoad.getExtremeElements(),
        c -> Durations.toMillis(c.getPackageMetricsInternal().getLoadDuration()));
    logIfNonEmpty(
        "Largest packages (num targets)",
        largestPackages.getExtremeElements(),
        c -> c.getPackageMetricsInternal().getNumTargets());
    logIfNonEmpty(
        "Packages with most computation steps",
        packagesWithMostComputationSteps.getExtremeElements(),
        c -> c.getPackageMetricsInternal().getComputationSteps());
    logIfNonEmpty(
        "Packages with most transitive loads (num bzl files)",
        packagesWithMostTransitiveLoads.getExtremeElements(),
        c -> c.getPackageMetricsInternal().getNumTransitiveLoads());
    clear();
  }

  @Override
  public Type getRecorderType() {
    return PackageMetricsRecorder.Type.ONLY_EXTREMES;
  }

  @Override
  public synchronized Collection<PackageMetrics> getPackageMetrics() {
    return Streams.concat(
            slowestPackagesToLoad.getExtremeElements().stream(),
            packagesWithMostComputationSteps.getExtremeElements().stream(),
            largestPackages.getExtremeElements().stream(),
            packagesWithMostTransitiveLoads.getExtremeElements().stream())
        .map(c -> c.getPackageMetrics())
        .collect(toImmutableSet());
  }

  private static void logIfNonEmpty(
      String logLinePrefix,
      List<PackageMetricsContainer> extremeElements,
      Function<PackageMetricsContainer, Long> valueMapper) {
    List<String> logString =
        extremeElements.stream()
            .map(v -> String.format("%s (%d)", v.getPackageIdentifier(), valueMapper.apply(v)))
            .collect(toImmutableList());
    if (!extremeElements.isEmpty()) {
      logger.atInfo().log("%s: %s", logLinePrefix, Joiner.on(", ").join(logString));
    }
  }
}
