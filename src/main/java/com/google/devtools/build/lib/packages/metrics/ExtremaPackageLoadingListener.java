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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.Extrema;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.annotation.concurrent.GuardedBy;

/** Tracks per-invocation extreme package loading events. */
public class ExtremaPackageLoadingListener implements PackageLoadingListener {
  @GuardedBy("this")
  private int currentNumPackagesToTrack;

  @GuardedBy("this")
  private Extrema<PackageIdentifierAndLong> slowestPackagesToLoad;

  @GuardedBy("this")
  private Extrema<PackageIdentifierAndLong> largestPackages;

  @GuardedBy("this")
  private Extrema<PackageIdentifierAndLong> packagesWithMostTransitiveLoads;

  @GuardedBy("this")
  private Extrema<PackageIdentifierAndLong> packagesWithMostComputationSteps;

  private static ExtremaPackageLoadingListener instance = null;

  public static synchronized ExtremaPackageLoadingListener getInstance() {
    if (instance == null) {
      instance = new ExtremaPackageLoadingListener();
    }
    return instance;
  }

  private ExtremaPackageLoadingListener() {
    this.currentNumPackagesToTrack = 0;
    this.slowestPackagesToLoad = Extrema.max(0);
    this.largestPackages = Extrema.max(0);
    this.packagesWithMostTransitiveLoads = Extrema.max(0);
    this.packagesWithMostComputationSteps = Extrema.max(0);
  }

  synchronized void setNumPackagesToTrack(int numPackagesToTrack) {
    Preconditions.checkArgument(numPackagesToTrack >= 0, "num packages must be >= 0");
    if (numPackagesToTrack == currentNumPackagesToTrack) {
      // Micro-optimization to avoid turning over collections.
      clear();
      return;
    }

    currentNumPackagesToTrack = numPackagesToTrack;
    this.slowestPackagesToLoad = Extrema.max(currentNumPackagesToTrack);
    this.largestPackages = Extrema.max(currentNumPackagesToTrack);
    this.packagesWithMostTransitiveLoads = Extrema.max(currentNumPackagesToTrack);
    this.packagesWithMostComputationSteps = Extrema.max(currentNumPackagesToTrack);
  }

  @Override
  public synchronized void onLoadingCompleteAndSuccessful(
      Package pkg, StarlarkSemantics starlarkSemantics, long loadTimeNanos) {
    if (currentNumPackagesToTrack == 0) {
      // Micro-optimization - no need to track.
      return;
    }

    long loadTimeMillis = TimeUnit.MILLISECONDS.convert(loadTimeNanos, TimeUnit.NANOSECONDS);
    slowestPackagesToLoad.aggregate(
        new PackageIdentifierAndLong(pkg.getPackageIdentifier(), loadTimeMillis));

    packagesWithMostComputationSteps.aggregate(
        new PackageIdentifierAndLong(pkg.getPackageIdentifier(), pkg.getComputationSteps()));

    largestPackages.aggregate(
        new PackageIdentifierAndLong(pkg.getPackageIdentifier(), pkg.getTargets().size()));

    packagesWithMostTransitiveLoads.aggregate(
        new PackageIdentifierAndLong(
            pkg.getPackageIdentifier(), pkg.getStarlarkFileDependencies().size()));
  }

  public synchronized TopPackages getTopPackages() {
    TopPackages result =
        new TopPackages(
            slowestPackagesToLoad.getExtremeElements(),
            packagesWithMostComputationSteps.getExtremeElements(),
            largestPackages.getExtremeElements(),
            packagesWithMostTransitiveLoads.getExtremeElements());
    return result;
  }

  synchronized TopPackages getAndResetTopPackages() {
    TopPackages result = getTopPackages();
    clear();
    return result;
  }

  private synchronized void clear() {
    slowestPackagesToLoad.clear();
    packagesWithMostComputationSteps.clear();
    largestPackages.clear();
    packagesWithMostTransitiveLoads.clear();
  }

  /** Container around lists of packages which are slow or large in some form. */
  public static class TopPackages {
    private final List<PackageIdentifierAndLong> slowestPackages;
    private final List<PackageIdentifierAndLong> packagesWithMostComputationSteps;
    private final List<PackageIdentifierAndLong> largestPackages;
    private final List<PackageIdentifierAndLong> packagesWithMostTransitiveLoads;

    private TopPackages(
        List<PackageIdentifierAndLong> slowestPackages,
        List<PackageIdentifierAndLong> packagesWithMostComputationSteps,
        List<PackageIdentifierAndLong> largestPackages,
        List<PackageIdentifierAndLong> packagesWithMostTransitiveLoads) {
      this.slowestPackages = slowestPackages;
      this.packagesWithMostComputationSteps = packagesWithMostComputationSteps;
      this.largestPackages = largestPackages;
      this.packagesWithMostTransitiveLoads = packagesWithMostTransitiveLoads;
    }

    public List<PackageIdentifierAndLong> getSlowestPackages() {
      return slowestPackages;
    }

    public List<PackageIdentifierAndLong> getPackagesWithMostComputationSteps() {
      return packagesWithMostComputationSteps;
    }

    public List<PackageIdentifierAndLong> getLargestPackages() {
      return largestPackages;
    }

    public List<PackageIdentifierAndLong> getPackagesWithMostTransitiveLoads() {
      return packagesWithMostTransitiveLoads;
    }
  }

  /** A pair of PackageIdentifier and a corresponding value. */
  public static class PackageIdentifierAndLong implements Comparable<PackageIdentifierAndLong> {
    private final PackageIdentifier pkgId;
    private final long val;

    private PackageIdentifierAndLong(PackageIdentifier pkgId, long val) {
      this.pkgId = pkgId;
      this.val = val;
    }

    public long getVal() {
      return val;
    }

    public PackageIdentifier getPkgId() {
      return pkgId;
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
