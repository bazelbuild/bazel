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

package com.google.devtools.build.lib.unix;

import com.google.common.annotations.VisibleForTesting;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 * Parse and return information from /proc/meminfo. In case of duplicate entries the first one is
 * used and other values are skipped.
 */
public class ProcMeminfoParser {

  public static final String FILE = "/proc/meminfo";

  private final byte[] content;
  private final int length;

  /**
   * Populates memory information by reading /proc/meminfo.
   * @throws IOException if reading the file failed.
   */
  public ProcMeminfoParser() throws IOException {
    this(FILE);
  }

  @VisibleForTesting
  public ProcMeminfoParser(String fileName) throws IOException {
    byte[] buffer = new byte[4096];
    int length = 0;
    try (InputStream in = new FileInputStream(fileName)) {
      int read;
      while ((read = in.read(buffer, length, buffer.length - length)) != -1) {
        length += read;
        if (length == buffer.length) {
          buffer = Arrays.copyOf(buffer, 2 * buffer.length);
        }
      }
    }
    this.content = buffer;
    this.length = length;
  }

  /**
   * Returns the value of the first line for the given keyword with a valid numeric value, or -1 if
   * there is no such line. Non-digit characters in the value, such as the unit, are ignored.
   */
  private long findKb(String keyword) {
    int lineStart = 0;
    while (lineStart < length) {
      int lineEnd = lineStart;
      while (lineEnd < length && content[lineEnd] != '\n' && content[lineEnd] != '\r') {
        lineEnd++;
      }
      if (lineHasKeyword(lineStart, lineEnd, keyword)) {
        long value = parseDigits(lineStart + keyword.length() + 1, lineEnd);
        if (value != -1) {
          return value;
        }
      }
      lineStart = lineEnd + 1;
    }
    return -1;
  }

  private boolean lineHasKeyword(int lineStart, int lineEnd, String keyword) {
    int colon = lineStart + keyword.length();
    if (colon >= lineEnd || content[colon] != ':') {
      return false;
    }
    for (int i = 0; i < keyword.length(); i++) {
      if (content[lineStart + i] != keyword.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /** Parses the digits in the given range, returning -1 if there are none or they overflow. */
  private long parseDigits(int start, int end) {
    long value = -1;
    for (int i = start; i < end; i++) {
      int digit = content[i] - '0';
      if (digit < 0 || digit > 9) {
        continue;
      }
      if (value == -1) {
        value = 0;
      }
      if (value > (Long.MAX_VALUE - digit) / 10) {
        return -1;
      }
      value = value * 10 + digit;
    }
    return value;
  }

  /** Gets a named field in KB. */
  long getRamKb(String keyword) throws KeywordNotFoundException {
    long value = findKb(keyword);
    if (value == -1) {
      throw new KeywordNotFoundException(keyword);
    }
    return value;
  }

  /** Return the total physical memory. */
  public long getTotalKb() throws KeywordNotFoundException {
    return getRamKb("MemTotal");
  }

  /**
   * Convert KB to MB.
   */
  public static double kbToMb(long kb) {
    return kb >> 10;
  }

  /**
   * Reads the amount of *available* memory as reported by the kernel. See https://goo.gl/ABn283 for
   * why this is better than trying to figure it out ourselves. This corresponds to the MemAvailable
   * line in /proc/meminfo.
   */
  public long getFreeRamKb() throws KeywordNotFoundException {
    long memAvailable = findKb("MemAvailable");
    if (memAvailable != -1) {
      return memAvailable;
    }
    // We have no MemAvailable in /proc/meminfo; fall back to the previous estimation.
    return getRamKb("MemTotal")
        - getRamKb("Active")
        // Blaze doesn't want to use more than a third of inactive ram...
        - (long) (getRamKb("Inactive") * 0.3)
        // ...and doesn't want to assume more than 80% of the slab memory can be reallocated.
        - (long) (getRamKb("Slab") * 0.8);
    // That said, this estimate will be more inaccurate as it diverges from kernel internals.
  }

  /** Exception thrown when /proc/meminfo does not have a requested key. Should be tolerated. */
  public static class KeywordNotFoundException extends IOException {
    private KeywordNotFoundException(String keyword) {
      super("Can't locate " + keyword + " in the /proc/meminfo");
    }
  }
}
