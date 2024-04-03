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

import com.sun.management.OperatingSystemMXBean;
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
}
