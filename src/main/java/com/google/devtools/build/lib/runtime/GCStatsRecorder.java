// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.lang.management.GarbageCollectorMXBean;
import java.util.ArrayList;
import java.util.List;

/**
 * Record GC stats for a build.
 */
public class GCStatsRecorder {

  private final Iterable<GarbageCollectorMXBean> mxBeans;
  private final ImmutableMap<String, GCStat> initialData;

  public GCStatsRecorder(Iterable<GarbageCollectorMXBean> mxBeans) {
    this.mxBeans = mxBeans;
    ImmutableMap.Builder<String, GCStat> initialData = ImmutableMap.builder();
    for (GarbageCollectorMXBean mxBean : mxBeans) {
      String name = mxBean.getName();
      initialData.put(name, new GCStat(name, mxBean.getCollectionCount(),
          mxBean.getCollectionTime()));
    }
    this.initialData = initialData.build();
  }

  public Iterable<GCStat> getCurrentGcStats() {
    List<GCStat> stats = new ArrayList<>();
    for (GarbageCollectorMXBean mxBean : mxBeans) {
      String name = mxBean.getName();
      GCStat initStat = Preconditions.checkNotNull(initialData.get(name));
      stats.add(new GCStat(name,
          mxBean.getCollectionCount() - initStat.getNumCollections(),
          mxBean.getCollectionTime() - initStat.getTotalTimeInMs()));
    }
    return stats;
  }

  /** Represents the garbage collections statistics for one collector (For example CMS). */
  public static class GCStat {

    private final String name;
    private final long numCollections;
    private final long totalTimeInMs;

    public GCStat(String name, long numCollections, long totalTimeInMs) {
      this.name = name;
      this.numCollections = numCollections;
      this.totalTimeInMs = totalTimeInMs;
    }

    /** Name of the Collector. For example CMS. */
    public String getName() { return name; }

    /** Number of invocations for a build. */
    public long getNumCollections() { return numCollections; }

    /**
     * Total time spend in GC for the collector. Note that the time does need to be exclusive (aka a
     * stop-the-world GC).
     */
    public long getTotalTimeInMs() { return totalTimeInMs; }

    @Override
    public String toString() {
      return "GC time for '" + name + "' collector: " + numCollections
          + " collections using " + totalTimeInMs + "ms";
    }
  }
}
