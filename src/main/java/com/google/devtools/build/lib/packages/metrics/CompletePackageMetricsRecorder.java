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
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.protobuf.Duration;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.concurrent.GuardedBy;

/** PackageMetricsRecorder that records all available metrics for all package loads. */
final class CompletePackageMetricsRecorder implements PackageMetricsRecorder {

  @GuardedBy("this")
  private final HashMap<PackageIdentifier, PackageLoadMetrics> metrics = new HashMap<>();

  CompletePackageMetricsRecorder() {}

  @Override
  public synchronized void recordMetrics(PackageIdentifier pkgId, PackageLoadMetrics metrics) {
    this.metrics.put(pkgId, metrics);
  }

  @Override
  public synchronized Map<PackageIdentifier, Duration> getLoadTimes() {
    return Maps.transformValues(metrics, PackageLoadMetrics::getLoadDuration);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getComputationSteps() {
    return Maps.transformValues(metrics, PackageLoadMetrics::getComputationSteps);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTargets() {
    return Maps.transformValues(metrics, PackageLoadMetrics::getNumTargets);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getNumTransitiveLoads() {
    return Maps.transformValues(metrics, PackageLoadMetrics::getNumTransitiveLoads);
  }

  @Override
  public synchronized Map<PackageIdentifier, Long> getPackageOverhead() {
    return Maps.transformValues(
        Maps.filterValues(metrics, PackageLoadMetrics::hasPackageOverhead),
        PackageLoadMetrics::getPackageOverhead);
  }

  @Override
  public synchronized void clear() {
    metrics.clear();
  }

  @Override
  public void loadingFinished() {
    clear();
  }

  @Override
  public Type getRecorderType() {
    return PackageMetricsRecorder.Type.ALL;
  }

  @Override
  public synchronized ImmutableCollection<PackageLoadMetrics> getPackageLoadMetrics() {
    // lazily set the pkgName when requested.
    return metrics.entrySet().stream()
        .map(e -> e.getValue().toBuilder().setName(e.getKey().toString()).build())
        .collect(toImmutableList());
  }
}
