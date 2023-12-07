// Copyright 2016 The Bazel Authors. All rights reserved.
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

/** This class computes the local host's resource capacity. */
final class LocalHostResource {

  private static final ResourceSet DEFAULT_RESOURCES =
      ResourceSet.create(
          // Only com.sun.management.OperatingSystemMXBean provides the total physical memory size.
          // This bean is container-aware as of JDK 14.
          // https://github.com/openjdk/jdk/commit/7b82266a159ce363708e347aba7e1b0d38206b48
          ((OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean())
                  .getTotalPhysicalMemorySize()
              / (1024.0 * 1024.0),
          // As of JDK 11, availableProcessors is aware of cgroups as commonly used by containers.
          // https://hg.openjdk.java.net/jdk/hs/rev/7f22774a5f42#l6.178
          Runtime.getRuntime().availableProcessors(),
          Integer.MAX_VALUE);

  public static ResourceSet get() {
    return DEFAULT_RESOURCES;
  }

  private LocalHostResource() {}
}
