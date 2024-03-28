// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.metrics;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.sandbox.Cgroup;
import java.util.Map;

/** Collects resource usage of processes from their cgroups {@code CgroupsInfo}. */
public class CgroupsInfoCollector {
  // TODO(b/292634407, b/323341972): Extract a common interface between CgroupsInfoCollector and
  //  PsInfoCollector to be passed to the WorkerProcessMetricsCollector. Then make both classes
  //  final.

  // Mainly a singleton for mocking purposes. But also useful if we want to persist additional
  // state as in the PsInfoCollector.
  private static final CgroupsInfoCollector instance = new CgroupsInfoCollector();

  private CgroupsInfoCollector() {}

  public static CgroupsInfoCollector instance() {
    return instance;
  }

  public ResourceSnapshot collectResourceUsage(Map<Long, Cgroup> pidToCgroups, Clock clock) {
    ImmutableMap.Builder<Long, Integer> pidToMemoryInKb = ImmutableMap.builder();

    for (Map.Entry<Long, Cgroup> entry : pidToCgroups.entrySet()) {
      Cgroup cgroup = entry.getValue();
      // TODO(b/292634407): Consider how to handle the unlikely case where only some cgroups are
      //  invalid.
      if (cgroup.exists()) {
        pidToMemoryInKb.put(entry.getKey(), entry.getValue().getMemoryUsageInKb());
      }
    }
    return ResourceSnapshot.create(pidToMemoryInKb.buildOrThrow(), clock.now());
  }
}
