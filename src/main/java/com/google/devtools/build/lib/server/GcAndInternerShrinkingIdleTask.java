// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.concurrent.PooledInterner;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.util.StringUtilities;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.time.Duration;

/** An {@link IdleTask} to run the garbage collector and shrink interner pools. */
public final class GcAndInternerShrinkingIdleTask implements IdleTask {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final boolean stateKeptAfterBuild;

  public GcAndInternerShrinkingIdleTask(boolean stateKeptAfterBuild) {
    this.stateKeptAfterBuild = stateKeptAfterBuild;
  }

  @Override
  public String displayName() {
    return "GC and interner shrinking";
  }

  @Override
  public Duration delay() {
    // If state was kept after the build, wait for a few seconds before triggering GC, to
    // avoid unnecessarily slowing down an immediately following incremental build.
    return stateKeptAfterBuild ? Duration.ofSeconds(10) : Duration.ZERO;
  }

  @Override
  public void run() {
    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
    MemoryUsage before = memBean.getHeapMemoryUsage();
    try (var p = GoogleAutoProfilerUtils.logged("Idle GC")) {
      System.gc();
    }
    // Shrinking interner pools can take multiple seconds for large builds, and is maximally
    // effective for builds that don't keep state. Avoid running it if state was kept, or if the
    // cleanup was interrupted in any case.
    if (!Thread.interrupted() && !stateKeptAfterBuild) {
      try (var p = GoogleAutoProfilerUtils.logged("Idle interner shrinking")) {
        PooledInterner.shrinkAll();
      }
    }
    MemoryUsage after = memBean.getHeapMemoryUsage();

    logger.atInfo().log(
        "[Idle GC] used: %s -> %s, committed: %s -> %s",
        StringUtilities.prettyPrintBytes(before.getUsed()),
        StringUtilities.prettyPrintBytes(after.getUsed()),
        StringUtilities.prettyPrintBytes(before.getCommitted()),
        StringUtilities.prettyPrintBytes(after.getCommitted()));
  }
}
