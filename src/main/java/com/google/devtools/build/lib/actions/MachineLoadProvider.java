// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.OS;
import com.sun.management.OperatingSystemMXBean;
import java.io.IOException;
import java.lang.management.ManagementFactory;

/** A provider that collects the load of a machine for the resource manager. */
public class MachineLoadProvider {

  // Operating system bean used to collect statistic about CPU load of system.
  private static final OperatingSystemMXBean osBean =
      (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

  private static class Singleton {
    static final MachineLoadProvider instance = new MachineLoadProvider();
  }

  /** Returns singleton instance of the machine load provider. */
  public static MachineLoadProvider instance() {
    return Singleton.instance;
  }

  private MachineLoadProvider() {}

  /** Returns "recent" CPU load of the machine as number between 0 and number of cores. */
  public double getCurrentCpuUsage() {
    double cpuLoad = osBean.getCpuLoad();
    int numProcessors = Runtime.getRuntime().availableProcessors();
    return cpuLoad * numProcessors;
  }

  /** Returns current memory usage of the machine in MB. */
  public double getCurrentMemoryUsageMb() {
    long systemMemoryUsageMb = -1;
    if (OS.getCurrent() == OS.LINUX) {
      // On Linux we get a better estimate by using /proc/meminfo. See
      // https://www.linuxatemyram.com/ for more info on buffer caches.
      try {
        ProcMeminfoParser procMeminfoParser = new ProcMeminfoParser("/proc/meminfo");
        systemMemoryUsageMb =
            (procMeminfoParser.getTotalKb() - procMeminfoParser.getFreeRamKb()) / 1024;
      } catch (IOException e) {
        // Silently ignore and fallback.
      }
    }
    if (systemMemoryUsageMb < 0) {
      // In case we aren't running on Linux or reading /proc/meminfo failed, fall back to
      // the OS bean.
      systemMemoryUsageMb =
          (osBean.getTotalMemorySize() - osBean.getFreeMemorySize()) / (1024 * 1024);
    }
    return systemMemoryUsageMb;
  }
}
