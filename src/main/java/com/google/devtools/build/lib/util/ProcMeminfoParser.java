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

package com.google.devtools.build.lib.util;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;

/**
 * Parse and return information from /proc/meminfo.
 */
public class ProcMeminfoParser {

  public static final String FILE = "/proc/meminfo";

  private final Map<String, Long> memInfo;

  /**
   * Populates memory information by reading /proc/meminfo.
   * @throws IOException if reading the file failed.
   */
  public ProcMeminfoParser() throws IOException {
    this(FILE);
  }

  @VisibleForTesting
  public ProcMeminfoParser(String fileName) throws IOException {
    List<String> lines = Files.readLines(new File(fileName), Charset.defaultCharset());
    ImmutableMap.Builder<String, Long> builder = ImmutableMap.builder();
    for (String line : lines) {
      int colon = line.indexOf(':');
      if (colon == -1) {
        continue;
      }
      String keyword = line.substring(0, colon);
      String valString = line.substring(colon + 1);
      try {
        long val =  Long.parseLong(CharMatcher.inRange('0', '9').retainFrom(valString));
        builder.put(keyword, val);
      } catch (NumberFormatException e) {
        // Ignore: we'll fail later if somebody tries to capture this value.
      }
    }
    memInfo = builder.build();
  }

  /**
   * Gets a named field in KB.
   */
  public long getRamKb(String keyword) {
    Long val = memInfo.get(keyword);
    if (val == null) {
      throw new IllegalArgumentException("Can't locate " + keyword + " in the /proc/meminfo");
    }
    return val;
  }

  /**
   * Return the total physical memory.
   */
  public long getTotalKb() {
    return getRamKb("MemTotal");
  }

  /**
   * Return the inactive memory.
   */
  public long getInactiveKb() {
    return getRamKb("Inactive");
  }

  /**
   * Return the active memory.
   */
  public long getActiveKb() {
    return getRamKb("Active");
  }

  /**
   * Return the slab memory.
   */
  public long getSlabKb() {
    return getRamKb("Slab");
  }

  /**
   * Convert KB to MB.
   */
  public static double kbToMb(long kb) {
    return kb / 1E3;
  }

  /**
   * Calculates amount of free RAM from /proc/meminfo content by using
   * MemTotal - (Active + 0.3*InActive + 0.8*Slab) formula.
   * Assumption here is that we allow Blaze to use all memory except when
   * used by active pages, 30% of the inactive pages (since they may become
   * active at any time) and 80% of memory used by kernel slab heap (since we
   * want to keep most of the slab heap in the memory but do not want it to
   * consume all available free memory).
   */
  public long getFreeRamKb() {
    return getTotalKb() - getActiveKb() - (long)(getInactiveKb() * 0.3) - (long)(getSlabKb() * 0.8);
  }
}
