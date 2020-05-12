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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.concurrent.TimeUnit;

/** Tracks per-invocation extreme package loading events. */
class ExtremaPackageLoadingListener implements PackageLoadingListener {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final int NUM_EXTREME_PACKAGES_TO_TRACK = 50;

  private static final ExtremaPackageLoadingListener instance = new ExtremaPackageLoadingListener();

  public static ExtremaPackageLoadingListener getInstance() {
    return instance;
  }

  private final Extrema<PackageIdentifierAndLong> slowestPackagesToLoad =
      Extrema.max(NUM_EXTREME_PACKAGES_TO_TRACK);
  private final Extrema<PackageIdentifierAndLong> largestPackages =
      Extrema.max(NUM_EXTREME_PACKAGES_TO_TRACK);
  private final Extrema<PackageIdentifierAndLong> packagesWithMostTransitiveLoads =
      Extrema.max(NUM_EXTREME_PACKAGES_TO_TRACK);
  private final Extrema<PackageIdentifierAndLong> packagesWithMostComputationSteps =
      Extrema.max(NUM_EXTREME_PACKAGES_TO_TRACK);

  private ExtremaPackageLoadingListener() {}

  @Override
  public void onLoadingCompleteAndSuccessful(
      Package pkg, StarlarkSemantics starlarkSemantics, long loadTimeNanos) {
    long loadTimeMillis = TimeUnit.MILLISECONDS.convert(loadTimeNanos, TimeUnit.NANOSECONDS);
    synchronized (slowestPackagesToLoad) {
      slowestPackagesToLoad.aggregate(
          new PackageIdentifierAndLong(pkg.getPackageIdentifier(), loadTimeMillis));
    }

    synchronized (packagesWithMostComputationSteps) {
      packagesWithMostComputationSteps.aggregate(
          new PackageIdentifierAndLong(pkg.getPackageIdentifier(), pkg.getComputationSteps()));
    }

    synchronized (largestPackages) {
      largestPackages.aggregate(
          new PackageIdentifierAndLong(pkg.getPackageIdentifier(), pkg.getTargets().size()));
    }

    synchronized (packagesWithMostTransitiveLoads) {
      packagesWithMostTransitiveLoads.aggregate(
          new PackageIdentifierAndLong(
              pkg.getPackageIdentifier(), pkg.getStarlarkFileDependencies().size()));
    }
  }

  public void logAndResetExtrema() {
    if (slowestPackagesToLoad.isEmpty()) {
      // One empty means they're all empty, skip logging.
      return;
    }
    maybeLogExtremaHelper("Slowest packages (ms)", slowestPackagesToLoad);
    maybeLogExtremaHelper("Largest packages (num targets)", largestPackages);
    maybeLogExtremaHelper("Packages with most computation steps", packagesWithMostComputationSteps);
    maybeLogExtremaHelper(
        "Packages with most transitive loads (num bzl files)", packagesWithMostTransitiveLoads);

    slowestPackagesToLoad.clear();
    packagesWithMostComputationSteps.clear();
    largestPackages.clear();
    packagesWithMostTransitiveLoads.clear();
  }

  private static void maybeLogExtremaHelper(
      String logLinePrefix, Extrema<PackageIdentifierAndLong> extrema) {
    ImmutableList<PackageIdentifierAndLong> extremeElements = extrema.getExtremeElements();
    if (!extremeElements.isEmpty()) {
      logger.atInfo().log(
          "%s: %s",
          logLinePrefix,
          Joiner.on(", ").join(extremeElements));
    }
  }

  private static class PackageIdentifierAndLong implements Comparable<PackageIdentifierAndLong> {
    private final PackageIdentifier pkgId;
    private final long val;

    private PackageIdentifierAndLong(PackageIdentifier pkgId, long val) {
      this.pkgId = pkgId;
      this.val = val;
    }

    @Override
    public int compareTo(PackageIdentifierAndLong other) {
      return Long.compare(val, other.val);
    }

    @Override
    public String toString() {
      return String.format("%s (%d)", pkgId, val);
    }
  }
}
