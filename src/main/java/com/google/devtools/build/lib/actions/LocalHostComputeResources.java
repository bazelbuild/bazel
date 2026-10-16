// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;

/** Provides information about the local host's hardware resources. */
@SuppressWarnings("NonFinalStaticField") // For testing.
public final class LocalHostComputeResources {
  private LocalHostComputeResources() {}

  private static double memoryMbOverride = -1.0;
  private static double cpuUsageOverride = -1.0;

  /** Returns the physical memory size in MB. */
  public static double getMemoryMb() {
    if (memoryMbOverride >= 0) {
      return memoryMbOverride;
    }
    // Only com.sun.management.OperatingSystemMXBean provides the total physical memory size.
    // This bean is container-aware as of JDK 14.
    // https://github.com/openjdk/jdk/commit/7b82266a159ce363708e347aba7e1b0d38206b48
    return ((OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean())
            .getTotalPhysicalMemorySize()
        / (1024.0 * 1024.0);
  }

  /** Returns the number of available processors. */
  public static double getCpuUsage() {
    if (cpuUsageOverride >= 0) {
      return cpuUsageOverride;
    }
    // As of JDK 11, availableProcessors is aware of cgroups as commonly used by containers.
    // https://hg.openjdk.java.net/jdk/hs/rev/7f22774a5f42#l6.178
    return Runtime.getRuntime().availableProcessors();
  }

  /**
   * Overrides hardware resources for testing.
   *
   * @param memoryMb The memory size in MB. Set to a negative value to clear the override.
   * @param cpuUsage The CPU usage. Set to a negative value to clear the override.
   */
  public static void setLocalHostComputeResourcesOverride(double memoryMb, double cpuUsage) {
    memoryMbOverride = memoryMb;
    cpuUsageOverride = cpuUsage;
  }

  /** Resets any hardware resource overrides set for testing. */
  public static void resetOverrides() {
    memoryMbOverride = -1.0;
    cpuUsageOverride = -1.0;
  }
}
