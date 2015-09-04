// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Splitter;
import com.google.common.io.Files;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.ProcMeminfoParser;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Set;

/**
 * This class estimates the local host's resource capacity.
 */
@ThreadCompatible
public final class LocalHostCapacity {

  /* If /proc/* information is not available, guess based on what the JVM thinks.  Anecdotally,
   * the JVM picks 0.22 the total available memory as maxMemory (tested on a standard Mac), so
   * multiply by 3, and divide by 2^20 because we want megabytes.
   */
  private static final ResourceSet DEFAULT_RESOURCES = ResourceSet.create(
      3.0 * (Runtime.getRuntime().maxMemory() >> 20),
      Runtime.getRuntime().availableProcessors(), 1.0,
      Integer.MAX_VALUE);

  private LocalHostCapacity() {}

  /**
   * Estimates of the local host's resource capacity,
   * obtained by reading /proc/cpuinfo and /proc/meminfo.
   */
  private static ResourceSet localHostCapacity;

  /**
   * Estimates of the local host's resource capacity,
   * obtained by reading /proc/cpuinfo and /proc/meminfo.
   */
  public static ResourceSet getLocalHostCapacity() {
    if (localHostCapacity == null) {
      localHostCapacity = getLocalHostCapacity("/proc/cpuinfo", "/proc/meminfo");
    }
    return localHostCapacity;
  }

  private static final Splitter NEWLINE_SPLITTER = Splitter.on('\n').omitEmptyStrings();

  @VisibleForTesting
  static int getLogicalCpuCount(String cpuinfoContent) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoContent);
    int count = 0;
    for (String line : lines) {
      if(line.startsWith("processor")) {
        count++;
      }
    }
    if (count == 0) {
      throw new IllegalArgumentException("Can't locate processor in the /proc/cpuinfo");
    }
    return count;
  }

  @VisibleForTesting
  static int getPhysicalCpuCount(String cpuinfoContent, int logicalCpuCount) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoContent);
    Set<String> uniq = new HashSet<>();
    for (String line : lines) {
      if(line.startsWith("physical id")) {
        uniq.add(line);
      }
    }
    int physicalCpuCount = uniq.size();
    if (physicalCpuCount == 0) {
      physicalCpuCount = logicalCpuCount;
    }
    return physicalCpuCount;
  }

  @VisibleForTesting
  static int getCoresPerCpu(String cpuinfoFileContent) {
    Iterable<String> lines = NEWLINE_SPLITTER.split(cpuinfoFileContent);
    Set<String> uniq = new HashSet<>();
    for (String line : lines) {
      if(line.startsWith("core id")) {
        uniq.add(line);
      }
    }
    int coresPerCpu = uniq.size();
    if (coresPerCpu == 0) {
      coresPerCpu = 1;
    }
    return coresPerCpu;
  }

  @VisibleForTesting
  static ResourceSet getLocalHostCapacity(String cpuinfoFile, String meminfoFile) {
    try {
      String cpuinfoContent = readContent(cpuinfoFile);
      ProcMeminfoParser memInfo = new ProcMeminfoParser(meminfoFile);
      int logicalCpuCount = getLogicalCpuCount(cpuinfoContent);
      int physicalCpuCount = getPhysicalCpuCount(cpuinfoContent, logicalCpuCount);
      int coresPerCpu = getCoresPerCpu(cpuinfoContent);
      int totalCores = coresPerCpu * physicalCpuCount;
      boolean hyperthreading = (logicalCpuCount != totalCores);
      double ramMb = ProcMeminfoParser.kbToMb(memInfo.getTotalKb());
      final double EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU = 0.6;
      return ResourceSet.create(
          ramMb,
          logicalCpuCount * (hyperthreading ? EFFECTIVE_CPUS_PER_HYPERTHREADED_CPU : 1.0),
          1.0,
          Integer.MAX_VALUE);
    } catch (IOException | IllegalArgumentException e) {
      return DEFAULT_RESOURCES;
    }
  }

  /**
   * For testing purposes only. Do not use it.
   */
  @VisibleForTesting
  static void setLocalHostCapacity(ResourceSet resources) {
    localHostCapacity = resources;
  }

  private static String readContent(String filename) throws IOException {
    return Files.toString(new File(filename), Charset.defaultCharset());
  }

}
