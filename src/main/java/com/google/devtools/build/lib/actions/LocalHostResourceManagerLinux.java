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

import com.google.common.base.Splitter;
import com.google.common.io.Files;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;

/**
 * This class estimates the local host's resource capacity for Linux.
 */
public class LocalHostResourceManagerLinux {
  private static String cpuInfoContent = null;

  private static final Splitter NEWLINE_SPLITTER = Splitter.on('\n').omitEmptyStrings();
  private static final String CPU_INFO_FILE = "/proc/cpuinfo";
  private static final String MEM_INFO_FILE = "/proc/meminfo";

  private static int getLogicalCpuCount() throws IOException {
    String content = getCpuInfoContent();
    return getLogicalCpuCountHelper(content);
  }

  private static double getMemoryInMb()
      throws IOException, ProcMeminfoParser.KeywordNotFoundException {
    return getMemoryInMbHelper(MEM_INFO_FILE);
  }

  public static ResourceSet getLocalHostResources() {
    try {
      int logicalCpuCount = getLogicalCpuCount();
      double ramMb = getMemoryInMb();

      return ResourceSet.create(
          ramMb,
          logicalCpuCount,
          1.0,
          Integer.MAX_VALUE);
    } catch (IOException | ProcMeminfoParser.KeywordNotFoundException e) {
      return null;
    }
  }

  private static String getCpuInfoContent() throws IOException {
    if (cpuInfoContent == null) {
      cpuInfoContent = readContent(CPU_INFO_FILE);
    }
    return cpuInfoContent;
  }

  private static String readContent(String filename) throws IOException {
    return Files.asCharSource(new File(filename), Charset.defaultCharset()).read();
  }

  /**
   * For testing purposes only. Do not use it.
   */
  public static int getLogicalCpuCountHelper(String content) throws IOException {
    int count = 0;
    Iterable<String> lines = NEWLINE_SPLITTER.split(content);
    for (String line : lines) {
      if (line.startsWith("processor")) {
        count++;
      }
    }
    if (count == 0) {
      throw new IllegalArgumentException("Can't get logical CPU count");
    }
    return count;
  }

  public static double getMemoryInMbHelper(String memInfoFileName)
      throws IOException, ProcMeminfoParser.KeywordNotFoundException {
    ProcMeminfoParser memInfo = new ProcMeminfoParser(memInfoFileName);
    double ramMb = ProcMeminfoParser.kbToMb(memInfo.getTotalKb());
    return ramMb;
  }
}
