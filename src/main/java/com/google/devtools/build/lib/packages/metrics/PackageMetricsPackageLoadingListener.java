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

import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.pkgcache.PackageOptions.LazyMacroExpansionPackages;
import com.google.protobuf.util.Durations;
import javax.annotation.concurrent.GuardedBy;
import net.starlark.java.eval.StarlarkSemantics;

/** Tracks per-invocation extreme package loading events. */
public class PackageMetricsPackageLoadingListener implements PackageLoadingListener {

  @GuardedBy("this")
  private PackageMetricsRecorder recorder;

  private boolean publishPackageMetricsInBep = false;

  @GuardedBy("PackageMetricsPackageLoadingListener.class")
  private static PackageMetricsPackageLoadingListener instance = null;

  public static synchronized PackageMetricsPackageLoadingListener getInstance() {
    if (instance == null) {
      instance = new PackageMetricsPackageLoadingListener();
    }
    return instance;
  }

  private PackageMetricsPackageLoadingListener() {
    this.recorder = null;
  }

  @Override
  public synchronized void onLoadingCompleteAndSuccessful(
      Package pkg,
      StarlarkSemantics starlarkSemantics,
      LazyMacroExpansionPackages lazyMacroExpansionPackages,
      Metrics metrics) {
    if (recorder == null) {
      // Micro-optimization - no need to track.
      return;
    }

    PackageLoadMetrics.Builder builder =
        PackageLoadMetrics.newBuilder()
            .setLoadDuration(Durations.fromNanos(metrics.loadTimeNanos()))
            .setGlobFilesystemOperationCost(metrics.globFilesystemOperationCost())
            .setComputationSteps(pkg.getComputationSteps())
            .setNumTargets(pkg.getTargets().size())
            .setNumTransitiveLoads(pkg.getDeclarations().countTransitivelyLoadedStarlarkFiles());

    if (pkg.getPackageOverhead().isPresent()) {
      builder.setPackageOverhead(pkg.getPackageOverhead().getAsLong());
    }

    recorder.recordMetrics(pkg.getPackageIdentifier(), builder.build());
  }

  /** Set the PackageMetricsRecorder for this listener. */
  public synchronized void setPackageMetricsRecorder(PackageMetricsRecorder recorder) {
    this.recorder = recorder;
  }

  public void setPublishPackageMetricsInBep(boolean publishPackageMetricsInBep) {
    this.publishPackageMetricsInBep = publishPackageMetricsInBep;
  }

  public boolean getPublishPackageMetricsInBep() {
    return publishPackageMetricsInBep;
  }

  /** Returns the PackageMetricsRecorder, if any, for the PackageLoadingListener. */
  public synchronized PackageMetricsRecorder getPackageMetricsRecorder() {
    return recorder;
  }
}
