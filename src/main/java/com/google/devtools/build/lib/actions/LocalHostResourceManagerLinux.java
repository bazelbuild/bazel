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

import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * This class estimates the local host's resource capacity for Linux.
 */
public class LocalHostResourceManagerLinux {

  private static final String MEM_INFO_FILE = "/proc/meminfo";

  private static int getLogicalCpuCount() throws IOException {
    // As of JDK 11, availableProcessors is aware of cgroups as commonly used by containers.
    // https://hg.openjdk.java.net/jdk/hs/rev/7f22774a5f42#l6.178
    return Runtime.getRuntime().availableProcessors();
  }

  private static double getMemoryInMb() throws IOException {
    return getMemoryInMbHelper(MEM_INFO_FILE);
  }

  @Nullable
  public static ResourceSet getLocalHostResources() {
    try {
      int logicalCpuCount = getLogicalCpuCount();
      double ramMb = getMemoryInMb();

      return ResourceSet.create(
          ramMb,
          logicalCpuCount,
          Integer.MAX_VALUE);
    } catch (IOException e) {
      return null;
    }
  }

  public static double getMemoryInMbHelper(String memInfoFileName) throws IOException {
    ProcMeminfoParser memInfo = new ProcMeminfoParser(memInfoFileName);
    double ramMb = ProcMeminfoParser.kbToMb(memInfo.getTotalKb());
    return ramMb;
  }
}
