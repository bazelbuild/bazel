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

package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.OS;

/**
 * This class estimates the local host's resource capacity.
 */
@ThreadCompatible
public final class LocalHostCapacity {
  private static final double EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU = 0.6;

  private static final OS currentOS = OS.getCurrent();
  private static ResourceSet localHostCapacity;

  private LocalHostCapacity() {}

  public static ResourceSet getLocalHostCapacity() {
    return getLocalHostCapacity(EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU);
  }

  public static ResourceSet getLocalHostCapacity(double cpusPerHyperthreadedCpuFactor) {
    if (localHostCapacity == null) {
      localHostCapacity = getNewLocalHostCapacity(cpusPerHyperthreadedCpuFactor);
    }
    return localHostCapacity;
  }

  private static ResourceSet getNewLocalHostCapacity(double cpusPerHyperthreadedCpuFactor) {
    ResourceSet localResources = null;
    switch (currentOS) {
      case DARWIN:
        localResources = LocalHostResourceManagerDarwin.getLocalHostResources(cpusPerHyperthreadedCpuFactor);
        break;
      case LINUX:
        localResources = LocalHostResourceManagerLinux.getLocalHostResources(cpusPerHyperthreadedCpuFactor);
        break;
      default:
        break;
    }
    if (localResources == null) {
      localResources = LocalHostResourceFallback.getLocalHostResources();
    }
    return localResources;
  }

  /**
   * Sets the local host capacity to hardcoded values.
   *
   * @param capacity the explicit capacity, or null to use the machine-specific values again
   */
  @VisibleForTesting
  public static void setLocalHostCapacity(ResourceSet capacity) {
    localHostCapacity = capacity;
  }
}
